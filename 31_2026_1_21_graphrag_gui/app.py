import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx

# ---------------------------------------------------------
# 1. 初期設定とセッション管理
# ---------------------------------------------------------
st.set_page_config(page_title="GraphRAG Visual Editor", layout="wide", page_icon="🕸️")

# グラフデータ（NetworkX）をセッションで保持
if 'graph' not in st.session_state:
    st.session_state['graph'] = nx.DiGraph()

# 接続操作用のステート（始点と終点を保持）
if 'source_node' not in st.session_state:
    st.session_state['source_node'] = None
if 'target_node' not in st.session_state:
    st.session_state['target_node'] = None

# ---------------------------------------------------------
# 2. サイドバー：ノード追加エリア
# ---------------------------------------------------------
st.sidebar.header("📦 ノード（要素）の追加")
new_node = st.sidebar.text_input("新しいノード名を入力", placeholder="例: ルミナイ株式会社")

if st.sidebar.button("➕ ノードを追加"):
    if new_node:
        if not st.session_state['graph'].has_node(new_node):
            st.session_state['graph'].add_node(new_node)
            st.sidebar.success(f"追加しました: {new_node}")
        else:
            st.sidebar.warning("そのノードは既に存在します。")
    else:
        st.sidebar.warning("ノード名を入力してください。")

st.sidebar.divider()

# グラフ情報の表示
st.sidebar.markdown(f"**現在の要素数:** {st.session_state['graph'].number_of_nodes()}")
st.sidebar.markdown(f"**現在の関係数:** {st.session_state['graph'].number_of_edges()}")

if st.sidebar.button("🗑️ 全データをリセット", type="primary"):
    st.session_state['graph'].clear()
    st.session_state['source_node'] = None
    st.session_state['target_node'] = None
    st.rerun()

# ---------------------------------------------------------
# 3. メインエリア：グラフ可視化とインタラクション
# ---------------------------------------------------------
st.title("🕸️ GraphRAG Visual Editor")
st.markdown("ノードをクリックして選択し、関係性を定義してください。")

col_graph, col_control = st.columns([3, 1])

with col_graph:
    # --- グラフデータの変換 (NetworkX -> Agraph) ---
    nodes = []
    edges = []

    # ノードのスタイル設定
    for n in st.session_state['graph'].nodes():
        # 選択中のノードは色を変える
        color = "#F7A7A6" # Default Pink
        if n == st.session_state['source_node']:
            color = "#5D5CDE" # Blue for Source
        elif n == st.session_state['target_node']:
            color = "#4CAF50" # Green for Target
            
        nodes.append(Node(id=n, label=n, size=25, color=color))

    # エッジのスタイル設定
    for u, v, d in st.session_state['graph'].edges(data=True):
        edges.append(Edge(source=u, target=v, label=d.get('relation', ''), type="CURVE_SMOOTH"))

    # グラフの設定（物理演算など）
    config = Config(
        width="100%", 
        height=500, 
        directed=True,
        nodeHighlightBehavior=True, 
        highlightColor="#F7A7A6",
        collapsible=False,
        physics=True,  # 物理演算を有効にしてフワフワ動くようにする
        hierarchical=False
    )

    # ★ここでグラフを描画し、クリックされたノードIDを取得★
    selected_node_id = agraph(nodes=nodes, edges=edges, config=config)

# ---------------------------------------------------------
# 4. コントロールパネル：選択と接続
# ---------------------------------------------------------
with col_control:
    st.subheader("🛠️ 接続操作")

    # グラフ上でノードがクリックされた時の処理
    if selected_node_id:
        st.info(f"選択中: **{selected_node_id}**")
        
        # 始点・終点の設定ボタン
        c1, c2 = st.columns(2)
        with c1:
            if st.button("始点に設定"):
                st.session_state['source_node'] = selected_node_id
                st.rerun()
        with c2:
            if st.button("終点に設定"):
                st.session_state['target_node'] = selected_node_id
                st.rerun()
    else:
        st.write("👈 グラフの丸をクリックしてください")

    st.divider()

    # 接続状況の表示
    src = st.session_state['source_node']
    tgt = st.session_state['target_node']

    st.write(f"**始点 (From):** {src if src else '未選択'}")
    st.write(f"**終点 (To):** {tgt if tgt else '未選択'}")

    # 両方選択されていたら、接続フォームを表示
    if src and tgt:
        if src == tgt:
            st.warning("自分自身には接続できません（今回は非対応）")
        else:
            relation_label = st.text_input("関係名 (例: 所属)", key="rel_input")
            
            if st.button("🔗 接続する (Connect)"):
                if relation_label:
                    st.session_state['graph'].add_edge(src, tgt, relation=relation_label)
                    # 接続したら選択解除
                    st.session_state['source_node'] = None
                    st.session_state['target_node'] = None
                    st.success(f"接続しました: {src} -> {tgt}")
                    st.rerun()
                else:
                    st.error("関係名を入力してください")

    # 選択解除ボタン
    if src or tgt:
        if st.button("選択クリア"):
            st.session_state['source_node'] = None
            st.session_state['target_node'] = None
            st.rerun()

# ---------------------------------------------------------
# 5. RAGとしての確認用（JSON出力）
# ---------------------------------------------------------
st.divider()
with st.expander("📊 生成されたグラフデータ (JSON形式)"):
    # グラフをJSONライクに表示して確認
    graph_data = nx.node_link_data(st.session_state['graph'])
    st.json(graph_data)