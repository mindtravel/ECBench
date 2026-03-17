# 运行所有模型处理完整ECBench数据

## 单行命令（处理所有数据）

```bash
cd /root/autodl-tmp/ECBench/evaluation && unset MAX_ITEMS && export OVERWRITE=true && bash run_all.sh
```

## 或者分步运行

```bash
cd /root/autodl-tmp/ECBench/evaluation

# 清除MAX_ITEMS限制，设置OVERWRITE以重新处理
unset MAX_ITEMS
export OVERWRITE=true

# 运行所有模型
bash run_all.sh
```

## 后台运行（推荐）

```bash
cd /root/autodl-tmp/ECBench/evaluation && unset MAX_ITEMS && export OVERWRITE=true && nohup bash run_all.sh > run_all.log 2>&1 &
```

查看进度：
```bash
tail -f /root/autodl-tmp/ECBench/evaluation/run_all.log
```

## 使用screen（推荐用于长时间运行）

```bash
screen -S ecbench
cd /root/autodl-tmp/ECBench/evaluation
unset MAX_ITEMS
export OVERWRITE=true
bash run_all.sh
# 按 Ctrl+A 然后 D 来detach，之后用 screen -r ecbench 恢复
```

## 说明

- `unset MAX_ITEMS`: 清除任何可能设置的MAX_ITEMS限制，确保处理所有4339条数据
- `export OVERWRITE=true`: 重新处理所有数据，即使之前已经有部分结果
- 运行时间：根据数据量，每个模型可能需要数小时
- 自动保存：每处理20个样本会自动保存，可以安全中断和恢复

## 输出文件

运行完成后，结果会保存在：
- InternVL2: `/root/autodl-tmp/ECBench/data/output_internvl2_pred.json`
- LongVA: `/root/autodl-tmp/ECBench/data/output_longva_pred.json`
