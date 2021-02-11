def split_data(x_data, y_data, ratio):
    assert len(x_data) > 0
    assert len(y_data) > 0
    assert len(x_data) == len(y_data)

    cutoff_idx = int(ratio * len(x_data))
    train_data_x = x_data[:cutoff_idx]
    test_data_x = x_data[cutoff_idx:]
    train_data_y = y_data[:cutoff_idx]
    test_data_y = y_data[cutoff_idx:]
    
    assert len(train_data_x) ==  len(train_data_y)
    assert len(test_data_x) == len(test_data_y)
    assert (len(train_data_x) + len(test_data_x)) ==  len(x_data)
    assert (len(train_data_y) + len(test_data_y)) == len(y_data)

    return train_data_x, train_data_y, test_data_x, test_data_y