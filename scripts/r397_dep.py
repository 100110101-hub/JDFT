import io

path = "qwb-core/crates/qwb-xc/Cargo.toml"
text = io.open(path, encoding="utf-8").read()
old = """[dependencies]
thiserror.workspace = true

[lints]"""
new = """[dependencies]
thiserror.workspace = true

[dev-dependencies]
# SC-06：逐点参考测试要从 `qwb-data/functional/*.yaml` 走**生产同一条**路径构造组件
# （`Registry::get_functional` → `build_term`），而不是在测试里复制 18 个参数常数。
qwb-registry.workspace = true

[lints]"""
assert old in text, "cargo anchor"
io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(old, new, 1))
print("dev-dependency added")
