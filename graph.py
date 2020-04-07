# ------------------------------------------------------------
# d3 helper functions
# ------------------------------------------------------------
def get_links(lst):
    "get adjacent pair"
    if len(lst) == 0:
        return None
    else:
        return list(zip(lst[:-1], lst[1:]))

def build_counter_links(lsts):
    "return counter object of all links"
    all_links = flatten([get_links(lst) for lst in lsts])
    return Counter(all_links)

def get_nodes_from_counter(cnts, limit):
    "get all unique nodes"
    lsts = cnts.most_common(limit)
    # list of tuples of links
    tup_lst = [lst[0] for lst in lsts]
    return list(set(flatten(tup_lst)))

def build_d3_node_lists(lst):
    return [{"name": x} for x in lst]
    
def build_d3_links(cnt, limit):
    lsts = cnt.most_common(limit)
    return [{"source": key[0], "target": key[1], "value": value} for (key, value) in lsts]

def append_prefix_to_list_items(lsts):
    "append idx to front of item"
    return [str(idx) + lst for idx, lst in enumerate(lsts)]

def generate_d3_dict(cnt, limit):
    "generate json file for d3 sankey"
    node_list = get_nodes_from_counter(cnt, limit)
    node_list = build_d3_node_lists(node_list)
    link_list = build_d3_links(cnt, limit)
    return {"links": link_list, "nodes": node_list}