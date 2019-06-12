def grouped_features(data):
    """
    This function creates some grouped features using the prices dataset.
    
    param data: Dataframe
    return data
    
    """
    data['competitor_price_per_prod'] = data['prod_id'].map(prices.groupby('prod_id')['competitor_price'].mean().to_dict())
    data['competitor_per_prod'] = data['prod_id'].map(prices.groupby('prod_id')['competitor'].count().to_dict())
    
    return data


def datetime_features(data):
    """
    This functions creates datetime features for the time series dataset.
    
    param data: time series data
    return data

    """
    data['sales_month'] = data.index.month
    data['sales_dayofweek'] = data.index.dayofweek
    data['sales_dayofyear'] = data.index.dayofyear
    data['sales_weekofyear'] = data.index.weekofyear
    
    return data

def difference_features(data, n):
    """
    This functions creates several features based on differences on quantity series.
    param data: time series data
    param n: number of features to create
    return: data
    
    """
    for s in range(1, n+1):
        data['quantity_diffs_{}'.format(s)] = data['qty_order'].diff(s)
    
    return data