import collections.abc

# https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
# Update a dictionary recursively so that any dicts contained within the dict are correctly updated
def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d