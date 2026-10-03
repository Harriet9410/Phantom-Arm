#!/usr/bin/env bash
# 一键给 FM9G4B-V 打 input_ids 重复传参补丁（换新实例后必做）
# 症状：TypeError: FM9GForCausalLM(...) got multiple values for keyword argument 'input_ids'
set -e
P=/root/inference/FM9G4B-V/modeling_fm9gv.py
[ -f "$P" ] || { echo "找不到 $P"; exit 1; }
cp -n "$P" "$P.bak_prepatch" 2>/dev/null || true
if grep -q '_input_ids_guard' "$P"; then echo "已经打过补丁，跳过"; else
python3 - "$P" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
ins = ('        kwargs.pop("_input_ids_guard", None)\n'
       '        for _k in ("input_ids", "position_ids", "inputs_embeds"):\n'
       '            kwargs.pop(_k, None)\n')
assert 'kwargs.pop("_input_ids_guard"' not in s, 'already patched'
i = s.index('return self.llm(')
io.open(p, 'w', encoding='utf-8').write(s[:i] + ins + s[i:])
print('PATCH OK')
PY
fi
rm -rf ~/.cache/huggingface/modules/transformers_modules /root/.cache/huggingface/modules/transformers_modules 2>/dev/null
echo "缓存已清。验证："
grep -q _input_ids_guard "$P" && echo "  PATCHED ✓" || echo "  补丁未生效 ✗"
