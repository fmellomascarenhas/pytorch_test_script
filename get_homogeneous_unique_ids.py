def create_unique_node_id(all_nodes):
      
    #creates unique_id
    all_nodes['unique_id'] = list(range(len(all_nodes)))
    #creates unique_id based on node type
    all_nodes['type_id'] = all_nodes.groupby(['node_type'])['node_id'].transform(lambda x: list(range(len(x))))

    return all_nodes

def merge_node_ids_and_edges(edges,
                             all_nodes,
                             merge_on_col,
                             cols_to_merge,
                             node_position):

    return pd.merge(edges,
                     all_nodes.rename(columns={col:col + f'_{node_position}' for col in merge_on_col + cols_to_merge}),
                     on=[col + f'_{node_position}' for col in merge_on_col],
                     how='left')       

############## merge all_nodes to edges ##############
all_nodes = create_unique_node_id(all_nodes)

merge_on_col = ['node_id']
cols_to_merge = [col for col in all_nodes.columns if col != merge_on_col]

edges = merge_node_ids_and_edges(edges,
                                 all_nodes,
                                 merge_on_col,
                                 cols_to_merge,
                                 node_position = '1')

edges = merge_node_ids_and_edges(edges,
                                 all_nodes,
                                 merge_on_col,
                                 cols_to_merge,
                                 node_position = '2')

########### CREATES GRAPH ###########
def add_to_g_dict(edges_df, g_dict):
    ntype1 = edges_df['node_type_1'].unique()[0]
    ntype2 = edges_df['node_type_2'].unique()[0]
    etype = edges_df['edge_type'].unique()[0]
    src = edges_df['type_id_1'].values
    dst = edges_df['type_id_2'].values
    g_dict.update({(ntype1, etype, ntype2): (torch.tensor(src), torch.tensor(dst))})

   
OBL_dict = {}
edges.groupby(['node_type_1', 'edge_type', 'node_type_2']).apply(add_to_g_dict, OBL_dict)

gOBL = dgl.heterograph(OBL_dict)
g = dgl.to_homogeneous(gOBL)

######### ADD HOMOGENEOUS NODES TO EDGES DF ############

map_node_type_id_to_str = {i: node_type for i, node_type in enumerate(gOBL.ntypes)}

data = torch.cat([g.ndata['_TYPE'].reshape(-1,1),
                  g.nodes().reshape(-1,1),
                  g.ndata['_ID'].reshape(-1,1)],
                  axis=-1).numpy()

df_homogeneous = pd.DataFrame(data, columns=['node_type', 'homo_unique_id', 'type_id'])
df_homogeneous['node_type'] = df_homogeneous['node_type'].transform(lambda x: map_node_type_id_to_str[x])

edges = merge_node_ids_and_edges(edges,
                              df_homogeneous,
                              merge_on_col = ['node_type', 'type_id'],
                              cols_to_merge = ['homo_unique_id'],
                              node_position = 1)

edges = merge_node_ids_and_edges(edges,
                              df_homogeneous,
                              merge_on_col = ['node_type', 'type_id'],
                              cols_to_merge = ['homo_unique_id'],
                              node_position = 2)

edges