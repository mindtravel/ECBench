# 运行所有模型处理ECBench

## 单行命令（推荐）

```bash
cd /root/autodl-tmp/ECBench/evaluation && bash run_all.sh
```

## 或者使用完整路径

```bash
bash /root/autodl-tmp/ECBench/evaluation/run_all.sh
```

## 后台运行（推荐，因为会运行很长时间）

```bash
cd /root/autodl-tmp/ECBench/evaluation && nohup bash run_all.sh > run_all.log 2>&1 &
```

查看进度：
```bash
tail -f /root/autodl-tmp/ECBench/evaluation/run_all.log
```

## 使用screen（推荐用于长时间运行）

```bash
screen -S ecbench
cd /root/autodl-tmp/ECBench/evaluation
bash run_all.sh
# 按 Ctrl+A 然后 D 来detach，之后用 screen -r ecbench 恢复
```

## 输出文件

运行完成后，结果会保存在：
- InternVL2: `/root/autodl-tmp/ECBench/data/output_internvl2_pred.json`
- LongVA: `/root/autodl-tmp/ECBench/data/output_longva_pred.json`

## 注意事项

1. 运行时间：根据数据量，每个模型可能需要数小时
2. 自动保存：每处理20个样本会自动保存，可以安全中断和恢复
3. 显存：确保GPU显存充足（建议至少24GB）
4. 如果某个模型失败，可以单独运行：
   ```bash
   bash run_internvl2.sh  # 只运行InternVL2
   bash run_longva.sh     # 只运行LongVA
   ```
