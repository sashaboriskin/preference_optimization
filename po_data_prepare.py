import kagglehub
import pandas as pd
import os
import seaborn as sns

DPO_DATASET_PATH = "data/dpo_dataset.csv"
KTO_DATASET_PATH = "data/kto_dataset.csv"
PROMPT_TEMPLATE = "Write a negative review on product with name "

def split_reviews_by_rating(df):
    result = {}
    for product_id, group in df.groupby('id'):
        low_rating_reviews = group[group['reviews.rating'] < 3]['reviews.text'].tolist()
        high_rating_reviews = group[group['reviews.rating'] > 3]['reviews.text'].tolist()
        result[product_id] = {
            'name': group['name'].iloc[0],
            'low_rating_reviews': low_rating_reviews,
            'high_rating_reviews': high_rating_reviews
        }
    return result

def main():

    path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")

    data = pd.concat(
        [
        pd.read_csv(os.path.join(path, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")),
        pd.read_csv(os.path.join(path, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")),
        pd.read_csv(os.path.join(path, "1429_1.csv"))
        ]
    )

    data = data[['id', 'name', 'reviews.doRecommend','reviews.rating', 'reviews.text']]
    data = data.drop_duplicates()
    data = data[(data['name'].isna()==False) & (data['reviews.rating'].isna()==False) & (data['reviews.text'].isna()==False)]
    data.reset_index(inplace=True, drop=True)


    reviews_by_product = split_reviews_by_rating(data)

    # Dpo data gen

    exclude = [k for k, p in reviews_by_product.items() if len(p["low_rating_reviews"])<28]


    all_pairs_by_product = {}
    for product in reviews_by_product.keys():
        if product not in exclude:
            all_pairs_by_product[reviews_by_product[product]['name']] = []
            j = 0
            for pos_review in reviews_by_product[product]['high_rating_reviews']:
                all_pairs_by_product[reviews_by_product[product]['name']].append([reviews_by_product[product]['low_rating_reviews'][j], pos_review])
                j = (j + 1) % len(reviews_by_product[product]['low_rating_reviews'])

    dpo_data = []
    for k, v in all_pairs_by_product.items():
        df = pd.DataFrame({
            "name": k,
            "items":v
        })
    dpo_data.append(df)

    dpo_data = pd.concat(dpo_data)

    dpo_data['prompt'] = dpo_data['name'].apply(lambda x: PROMPT_TEMPLATE + x)

    dpo_data['chosen'] = dpo_data['items'].apply(lambda x: x[0])
    dpo_data['rejected'] = dpo_data['items'].apply(lambda x: x[1])

    dpo_data[['prompt', 'chosen', 'rejected']].to_csv(DPO_DATASET_PATH, index=False)


    # Kto data gen

    kto_data = data[(data['reviews.rating'] != 3) & (~data['id'].isin(exclude))]

    kto_data['target'] = kto_data['reviews.rating'].apply(lambda x: True if x < 3 else False)

    kto_data = kto_data[['name', 'reviews.text', 'target']]

    kto_data = kto_data.rename(
        columns={
            "name": "product_name",
            "reviews.text": "text",
        }
    )

    kto_data['product_name'] = kto_data['product_name'].apply(lambda x: PROMPT_TEMPLATE + x)

    kto_data = kto_data.rename(
        columns={
            "product_name": "prompt",
            "text": "completion",
        }
    )

    kto_data.to_csv(KTO_DATASET_PATH, index=False)

if __name__ == '__main__':
    main()
