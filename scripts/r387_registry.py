import io
import re

# registry.yaml 里八个溶剂的 sha256 与数据仓版本要同步
PINS = {
    "water": "1ccc509ed9765e9655d6cdab802bb2c1ffa772c1e54305760541db19c79b016f",
    "methanol": "3c763aa993e8fd7aa0f76080751f2827f1b632bbbf172e8e89e01a51283bfd85",
    "ethanol": "62cb97f726f6948ae080bedd855fe4376e5e6ec0d281fc7aefb197a91adc9dec",
    "acetonitrile": "191c9a7caef1f1a71956c3d01f53025788fdd32863588065c74bf3ddccbbfe83",
    "dmso": "f68e8080b8fc9bf399302c39f16e030e0a3825890c5037b2b3587322bea96e15",
    "chloroform": "7a903ef58d0afc176188241bb8b6a590c8a368699574e1b1e6edeab4df6015e7",
    "toluene": "553dd3f7936592916fc7336e4f746dfb96a857f1b0bfd9825561543c1a8cd5de",
    "thf": "9a63add08e95a1a4b41264d8e62c8d23ed17c22e636825c88c7b52a484ca2444",
}

path = "qwb-data/registry.yaml"
text = io.open(path, encoding="utf-8").read()

for name, digest in PINS.items():
    pattern = re.compile(
        r"(- name: " + re.escape(name) + r"\n\s+path: solvent/" + re.escape(name) + r"\.yaml\n\s+sha256: )([0-9a-f]{64})"
    )
    replaced, count = pattern.subn(lambda m: m.group(1) + digest, text)
    assert count == 1, f"{name}: {count} matches"
    text = replaced

assert "version: 0.13.0" in text
text = text.replace("version: 0.13.0", "version: 0.14.0", 1)
io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("registry.yaml pins and version updated")
