# CoT Validation

## Planned Structure
```
cot-validation/
├── config/
├── data/
├── notebooks/
│   └── cot_validation.ipynb
├── src/
│   ├── __init__.py
│   ├── reasoning/
│   │   ├── inference.py
│   │   ├── formatting.py
│   │   └── certainty_calc.py     (the logic that calculates “by which token are we 90% sure”)
│   └── utils/
│   │   ├── data_handler.py
│   │   └── memory_management.py  (incl mem.py)
├── tests/
├── README.md
├── LICENSE
└── requirements.txt
```

