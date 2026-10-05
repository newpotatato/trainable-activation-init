# Код экспериментов

Код к статье «Начальные значения параметров обучаемых функций активации: чувствительность обучения, пробные прогоны и перенос значений».

## Файлы
| Файл | Что делает |
|---|---|
| `activations.py` | PReLU, Swish-β, ACON-C, SigLin-T; стандартные значения, кандидаты, диапазоны для случайной инициализации |
| `data.py` | загрузка и подготовка 8 задач (California Housing, Wine Quality, Covertype, Fashion-MNIST, Friedman #1–#3, синусоиды); кэш в `data/*.npz` |
| `train.py` | одно обучение: MLP 3×128 или 8×128, Adam 1e-3, 30 эпох. Стратегии: `default`, `random`, `pilot` (2 эп.), `pilot_d`, `pilot5`, `pilot_e<T>`, `transfer`, `default_long` (55 эп., равные затраты) |
| `run_all.py` | параллельный запуск сетки с дозаписью в `results/*.jsonl` (готовые запуски пропускаются) |
| `make_transfer.py` | значения для стратегии `transfer` (схема «оставить один набор данных») |
| `analyze.py` | подробная сводка `results/summary.md` |
| `figures.py` | рисунки `results/figures/fig1…fig7` и таблицы `results/article_tables.md` |
| `summary_vs_norm.py` | сводное сравнение пилота с нормой: `results/summary_vs_norm.md`, `fig8_vs_norm.png` |
| `build_article.py`, `write_article.py`, `docx_utils.py`, `docx_to_md.py` | сборка статьи `../article/*.docx` и её Markdown-версии |

Рисунки в статье и имена файлов соответствуют так:

| Рисунок в статье | Файл |
|---|---|
| Рис. 1 | `fig1_siglin_shape.png` |
| Рис. 2 | `fig2_improvement.png` |
| Рис. 3 | `fig4_curves.png` |
| Рис. 4 | `fig3_param_drift.png` |
| Рис. 5 | `fig6_smooth.png` |
| Рис. 6 | `fig7_pilot_length.png` |
| Рис. 7 | `fig5_deep.png` |
| Рис. 8 | `fig8_vs_norm.png` |

## Воспроизведение
```bash
pip install torch numpy scipy scikit-learn pandas openpyxl matplotlib python-docx
python data.py                                                     # скачать и подготовить данные

# основная серия, сеть 3×128 (360 обучений)
python run_all.py --workers 6 --datasets california wine covertype fashion
python make_transfer.py
python run_all.py --workers 6 --datasets california wine covertype fashion --strategies transfer
python run_all.py --workers 6 --acts aconc siglin --datasets california wine covertype fashion --strategies pilot_d

# гладкие задачи, 10 seed
python run_all.py --workers 6 --seeds 10 --datasets friedman friedman2 friedman3 sines --strategies default random pilot transfer
python run_all.py --workers 6 --seeds 10 --datasets friedman friedman2 friedman3 sines --strategies pilot5 --out results/runs_pilot5.jsonl
python run_all.py --workers 6 --seeds 10 --datasets friedman --acts swish aconc \
    --strategies pilot_e1 pilot_e2 pilot_e3 pilot_e8 --out results/runs_pilot_sweep.jsonl

# сводное сравнение с нормой: пилот 5 эп. на основных задачах и стандартная инициализация с равными затратами
python run_all.py --workers 4 --datasets california wine covertype fashion --strategies pilot5 --out results/runs_pilot5b.jsonl
python run_all.py --workers 4 --datasets covertype fashion --strategies default_long --out results/runs_default_long.jsonl
python run_all.py --workers 4 --datasets california wine --strategies default_long --out results/runs_default_long_cw.jsonl
python run_all.py --workers 4 --seeds 10 --datasets friedman friedman2 friedman3 sines --strategies default_long \
    --out results/runs_default_long_smooth.jsonl

# глубокая сеть 8×128
python run_all.py --workers 6 --arch mlp8 --acts prelu swish siglin --fixed relu sigmoid \
    --datasets california wine covertype fashion --strategies default transfer pilot --out results/runs_mlp8.jsonl
python run_all.py --workers 3 --arch mlp8 --acts prelu swish siglin --datasets california wine \
    --strategies pilot5 --out results/runs_pilot5_mlp8.jsonl

python analyze.py && python figures.py && python summary_vs_norm.py
python write_article.py && python docx_to_md.py                    # статья в ../article/
```

**Ресурсы.** Один процесс занимает до ~550 МБ памяти (Fashion-MNIST), поэтому число процессов выбирайте по свободной памяти. Всего 1805 обучений, на 6 ядрах CPU это несколько часов.

`results_v1/` — предварительный раунд (California, Adult и малый синтетический набор данных автора; другая стандартная инициализация SigLin-T). В статье упоминается только как предварительная серия.
