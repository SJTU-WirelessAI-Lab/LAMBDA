from __future__ import annotations

import argparse
import copy
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]
SF_ROOT = SERVER_ROOT / "region_files" / "San Francisco"
DEFAULT_OUTPUT_DIR = SF_ROOT / "Full SF"


def safe_prefix(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    text = "_".join(rel.parts)
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    return text.strip("_")


def matching_xmls(sf_root: Path, band: str, include_openground: bool) -> list[Path]:
    if band == "0-10G":
        pattern = "*_0-10G.xml"
    elif band == "10-100G":
        pattern = "*_10-100G.xml"
    else:
        raise ValueError(f"Unsupported band {band!r}")

    files = []
    for path in sorted(sf_root.rglob(pattern)):
        name = path.name.lower()
        if "rain" in name or "snow" in name or "fog" in name:
            continue
        if not include_openground and "openground" in name:
            continue
        if "full sf" in {part.lower() for part in path.parts}:
            continue
        files.append(path)
    return files


def rewrite_ids(element: ET.Element, prefix: str) -> None:
    for node in element.iter():
        if node.tag == "bsdf":
            continue
        if node.tag != "ref" and "id" in node.attrib:
            node.attrib["id"] = f"{prefix}__{node.attrib['id']}"
        if "name" in node.attrib and node.tag in {"shape", "integrator", "sensor", "emitter"}:
            node.attrib["name"] = f"{prefix}__{node.attrib['name']}"


def rewrite_refs(element: ET.Element, prefix: str, material_ids: set[str]) -> None:
    for node in element.iter("ref"):
        ref_id = node.attrib.get("id")
        if not ref_id:
            continue
        if node.attrib.get("name") == "bsdf" or ref_id in material_ids:
            continue
        node.attrib["id"] = f"{prefix}__{ref_id}"


def infer_radio_material_type(material_id: str) -> str:
    name = material_id.lower()
    if "glass" in name or "window" in name:
        return "glass"
    if "brick" in name:
        return "brick"
    if "metal" in name or "steel" in name or "alum" in name:
        return "metal"
    if "wood" in name:
        return "wood"
    if "marble" in name:
        return "marble"
    return "concrete"


def force_sionna_radio_material(bsdf: ET.Element) -> None:
    if bsdf.attrib.get("type") == "itu-radio-material":
        return
    material_id = bsdf.attrib.get("id", "")
    radio_type = infer_radio_material_type(material_id)
    bsdf.attrib["type"] = "itu-radio-material"
    for child in list(bsdf):
        bsdf.remove(child)
    ET.SubElement(bsdf, "string", {"name": "type", "value": radio_type})
    ET.SubElement(bsdf, "float", {"name": "thickness", "value": "0.1"})


def rewrite_filename_values(element: ET.Element, source_xml: Path, output_dir: Path) -> None:
    source_dir = source_xml.parent
    for node in element.iter("string"):
        if node.attrib.get("name") != "filename":
            continue
        filename = node.attrib.get("value")
        if not filename:
            continue
        mesh_path = (source_dir / filename).resolve()
        node.attrib["value"] = Path(os.path.relpath(mesh_path, output_dir.resolve())).as_posix()


def merge_xmls(xmls: list[Path], output_xml: Path, sf_root: Path) -> dict[str, int]:
    if not xmls:
        raise ValueError("No XML files selected for merge.")

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    first_root = ET.parse(xmls[0]).getroot()
    merged = ET.Element(first_root.tag, dict(first_root.attrib))
    if "version" not in merged.attrib:
        merged.attrib["version"] = "2.1.0"

    total_shapes = 0
    seen_bsdf_ids: set[str] = set()
    for source_xml in xmls:
        prefix = safe_prefix(source_xml, sf_root)
        root = ET.parse(source_xml).getroot()
        source_bsdf_ids = {child.attrib["id"] for child in list(root) if child.tag == "bsdf" and "id" in child.attrib}
        for child in list(root):
            if child.tag == "integrator" and any(existing.tag == "integrator" for existing in merged):
                continue
            copied = copy.deepcopy(child)
            if copied.tag == "bsdf":
                force_sionna_radio_material(copied)
                bsdf_id = copied.attrib.get("id", "")
                if bsdf_id in seen_bsdf_ids:
                    continue
                seen_bsdf_ids.add(bsdf_id)
                merged.append(copied)
                continue
            rewrite_ids(copied, prefix)
            rewrite_refs(copied, prefix, source_bsdf_ids)
            rewrite_filename_values(copied, source_xml, output_xml.parent)
            if copied.tag == "shape":
                total_shapes += 1
            merged.append(copied)

    tree = ET.ElementTree(merged)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return {"xml_count": len(xmls), "shape_count": total_shapes}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge San Francisco region XML tiles into full-scene XML files.")
    parser.add_argument("--sf-root", type=Path, default=SF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-openground", action="store_true", help="Include Square 3 openground XML tiles.")
    parser.add_argument("--band", choices=["0-10G", "10-100G", "both"], default="both")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bands = ["0-10G", "10-100G"] if args.band == "both" else [args.band]
    for band in bands:
        xmls = matching_xmls(args.sf_root, band, include_openground=args.include_openground)
        output_name = "fullsf_0-10G.xml" if band == "0-10G" else "fullsf_10-100G.xml"
        output_xml = args.output_dir / output_name
        stats = merge_xmls(xmls, output_xml, args.sf_root)
        print(f"[{band}] merged {stats['xml_count']} XML files, {stats['shape_count']} shapes")
        for xml in xmls:
            print(f"  - {xml.relative_to(args.sf_root)}")
        print(f"  -> {output_xml}")


if __name__ == "__main__":
    main()
