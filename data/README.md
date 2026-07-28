# AG News Dataset

This project uses the **AG News Classification Dataset**, a widely used benchmark dataset for text classification and Natural Language Processing (NLP) tasks.

## Dataset Overview

- **Training Samples:** 120,000
- **Testing Samples:** 7,600
- **Classes:** 4
  - 🌍 World
  - ⚽ Sports
  - 💼 Business
  - 🔬 Sci/Tech

Each record contains:
- **Class Index**
- **Title**
- **Description**

---

## Download Dataset

The dataset can be downloaded directly from Kaggle:

https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

After downloading, extract the files and place them inside the `data` folder as shown below.

```
data/
├── train.csv
└── test.csv
```

---

## Dataset Structure

```
train.csv
├── Class Index
├── Title
└── Description

test.csv
├── Class Index
├── Title
└── Description
```

---

## Citation

Zhang, Xiang, Junbo Zhao, and Yann LeCun.

**Character-level Convolutional Networks for Text Classification.**

Advances in Neural Information Processing Systems (NeurIPS), 2015.

---

## License

Please refer to the dataset license and terms of use on the Kaggle dataset page before redistributing or using the data for commercial purposes.
