# Raw data

The CSV files are not in this repository. The dataset is the Brazilian E-Commerce
Public Dataset by Olist. Get it from either place and put the CSV files in this folder:

- Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Olist's GitHub account: `git clone --depth 1 https://github.com/olist/work-at-olist-data`,
  then copy `datasets/*.csv` here

The pipeline reads these eight files:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_geolocation_dataset.csv`
- `product_category_name_translation.csv`

Olist publishes the dataset under CC BY-NC-SA 4.0. The model in `artifacts/` was trained
on it and is shared for demonstration only.
