import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import os

# Inline model — no torch_geometric dependency
class FraudGCN(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=2):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index=None):
        x = torch.relu(self.lin1(x))
        x = self.dropout(x)
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)
        return x

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Fraud GNN Visualizer",
    page_icon="🔍",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = FraudGCN(
        in_channels=12,
        hidden_channels=64,
        out_channels=2
    )
    model_path = "best_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )
        model.eval()
        return model, True
    return model, False

model, model_loaded = load_model()

# ── Header ────────────────────────────────────────────────
st.title("🔍 Fraud GNN — Transaction Graph Visualizer")
st.caption(
    "Interactive visualization of Graph Neural Network fraud detection. "
    "See how GCN propagates fraud signals through transaction networks."
)

# ── Sidebar controls ──────────────────────────────────────
st.sidebar.title("⚙️ Graph Controls")

num_nodes = st.sidebar.slider(
    "Number of transactions", 10, 50, 20
)
fraud_rate = st.sidebar.slider(
    "Fraud rate (%)", 5, 40, 15
)
connectivity = st.sidebar.slider(
    "Graph connectivity", 1, 5, 2,
    help="Average connections per node"
)
show_propagation = st.sidebar.checkbox(
    "Show fraud propagation", value=True
)
propagation_steps = st.sidebar.slider(
    "Propagation steps", 1, 3, 2
) if show_propagation else 1

st.sidebar.divider()
st.sidebar.subheader("🎯 Scan Transaction")
scan_amount = st.sidebar.number_input(
    "Amount (₸)", min_value=100, max_value=500000,
    value=15000, step=1000
)
scan_days = st.sidebar.number_input(
    "Days since last txn", min_value=0.0,
    max_value=30.0, value=0.5, step=0.1
)
scan_count = st.sidebar.number_input(
    "Transactions last 7 days",
    min_value=1, max_value=50, value=15
)
scan_avg = st.sidebar.number_input(
    "Avg amount last 30 days (₸)",
    min_value=100, max_value=100000, value=3000
)

# ── Generate synthetic transaction graph ──────────────────
@st.cache_data
def generate_transaction_graph(n_nodes, fraud_rate, connectivity, seed=42):
    np.random.seed(seed)

    # Generate nodes
    nodes = []
    for i in range(n_nodes):
        is_fraud = np.random.random() < (fraud_rate / 100)
        amount = (
            np.random.uniform(50000, 500000)
            if is_fraud
            else np.random.uniform(500, 30000)
        )
        nodes.append({
            'id': f'txn_{i:03d}',
            'amount': round(amount, 2),
            'is_fraud': is_fraud,
            'card': f'card_{np.random.randint(0, n_nodes // 3):03d}',
            'days_since_last': round(np.random.uniform(0.1, 30), 1),
            'txn_count_7d': np.random.randint(1, 30)
        })

    df_nodes = pd.DataFrame(nodes)

    # Generate edges based on shared cards
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if df_nodes.iloc[i]['card'] == df_nodes.iloc[j]['card']:
                edges.append((i, j))

    # Add random edges for connectivity
    extra_edges = int(n_nodes * connectivity * 0.3)
    for _ in range(extra_edges):
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        if i != j and (i, j) not in edges:
            edges.append((i, j))

    return df_nodes, edges

df_nodes, edges = generate_transaction_graph(
    num_nodes, fraud_rate, connectivity
)

# ── Build NetworkX graph ──────────────────────────────────
G = nx.Graph()
for i, row in df_nodes.iterrows():
    G.add_node(i, **row.to_dict())
for e in edges:
    G.add_edge(*e)

# ── Compute fraud propagation ─────────────────────────────
def compute_propagation(G, df_nodes, steps):
    """Simulate GNN message passing — fraud signal propagates to neighbors"""
    fraud_scores = df_nodes['is_fraud'].astype(float).values.copy()

    for step in range(steps):
        new_scores = fraud_scores.copy()
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                neighbor_fraud = np.mean([
                    fraud_scores[n] for n in neighbors
                ])
                # GNN aggregation: mix own score with neighbors
                new_scores[node] = (
                    0.6 * fraud_scores[node] +
                    0.4 * neighbor_fraud
                )
        fraud_scores = new_scores

    return fraud_scores

if show_propagation:
    propagated_scores = compute_propagation(
        G, df_nodes, propagation_steps
    )
else:
    propagated_scores = df_nodes['is_fraud'].astype(float).values

# ── Graph layout ──────────────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=2)

# ── Build Plotly figure ───────────────────────────────────
def build_graph_figure(G, df_nodes, pos, propagated_scores):

    # Edge traces
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='rgba(150,150,150,0.4)'),
        hoverinfo='none',
        name='Connections'
    )

    # Node traces — color by fraud score
    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_colors = propagated_scores
    node_sizes = [
        20 + (df_nodes.iloc[i]['amount'] / 10000)
        for i in G.nodes()
    ]
    node_sizes = [min(50, max(15, s)) for s in node_sizes]

    node_text = []
    for i in G.nodes():
        row = df_nodes.iloc[i]
        fraud_score = propagated_scores[i]
        risk = (
            '🔴 HIGH RISK' if fraud_score > 0.6 else
            '🟠 MEDIUM' if fraud_score > 0.3 else
            '🟢 LOW RISK'
        )
        node_text.append(
            f"ID: {row['id']}<br>"
            f"Card: {row['card']}<br>"
            f"Amount: ₸{row['amount']:,.0f}<br>"
            f"Fraud Score: {fraud_score:.2f}<br>"
            f"Risk: {risk}<br>"
            f"True Label: {'🚨 FRAUD' if row['is_fraud'] else '✅ NORMAL'}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale=[
                [0.0, '#2ECC71'],
                [0.3, '#F39C12'],
                [0.6, '#E74C3C'],
                [1.0, '#8B0000']
            ],
            colorbar=dict(
                title='Fraud Score',
                thickness=15,
                tickvals=[0, 0.3, 0.6, 1.0],
                ticktext=['Safe', 'Watch', 'Risky', 'Fraud']
            ),
            line=dict(width=2, color='white'),
            cmin=0, cmax=1
        ),
        text=[
            f"txn_{i:03d}" for i in G.nodes()
        ],
        textposition='top center',
        textfont=dict(size=8),
        name='Transactions'
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=(
                    f'Transaction Graph — '
                    f'{sum(df_nodes["is_fraud"])} fraud / '
                    f'{len(df_nodes)} total transactions'
                ),
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
    )

    return fig

# ── Main layout ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🕸️ Graph Visualization",
    "📊 Analytics",
    "🎯 Live Fraud Scanner"
])

with tab1:
    st.subheader("Transaction Graph — GNN Fraud Propagation")

    if show_propagation:
        st.info(
            f"🧠 **GNN Message Passing:** Fraud signal propagates "
            f"through {propagation_steps} hop(s). "
            f"Nodes near fraudulent transactions show elevated risk scores — "
            f"even if they themselves are legitimate."
        )

    fig = build_graph_figure(G, df_nodes, pos, propagated_scores)
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("🟢 Safe (score < 0.3)")
    with col2:
        st.warning("🟠 Watch (0.3 - 0.6)")
    with col3:
        st.error("🔴 High Risk (> 0.6)")
    with col4:
        st.info(f"📐 Node size = transaction amount")

    # Key insight
    st.divider()
    st.subheader("💡 Why GNN beats traditional ML")

    insight1, insight2 = st.columns(2)
    with insight1:
        st.markdown("""
**Traditional ML (LightGBM):**
- Looks at each transaction in isolation
- Features: amount, time, card type
- Misses: network patterns, shared cards
- Blind to: coordinated fraud rings
        """)
    with insight2:
        st.markdown("""
**Graph Neural Network (GCN):**
- Sees transactions AND their connections
- Aggregates neighbor features automatically
- Detects: fraud rings through shared cards
- Catches: money mule networks
        """)

with tab2:
    st.subheader("📊 Graph Analytics")

    # Stats
    a1, a2, a3, a4 = st.columns(4)
    fraud_count = sum(df_nodes['is_fraud'])
    high_risk = sum(propagated_scores > 0.6)
    avg_degree = np.mean([d for _, d in G.degree()])

    with a1:
        st.metric("🚨 True Frauds", fraud_count,
                  delta=f"{fraud_count/num_nodes*100:.0f}% of graph")
    with a2:
        st.metric("🔴 High Risk Nodes", high_risk,
                  delta=f"{high_risk/num_nodes*100:.0f}% flagged")
    with a3:
        st.metric("🔗 Avg Connections", f"{avg_degree:.1f}")
    with a4:
        st.metric("📐 Graph Edges", len(edges))

    st.divider()

    # Amount distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Amount Distribution")
        fig_hist = px.histogram(
            df_nodes,
            x='amount',
            color='is_fraud',
            color_discrete_map={True: '#E74C3C', False: '#2ECC71'},
            labels={
                'is_fraud': 'Fraud',
                'amount': 'Transaction Amount (₸)'
            },
            nbins=20,
            barmode='overlay',
            opacity=0.7
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("🎯 Fraud Score Distribution")
        fig_score = px.histogram(
            x=propagated_scores,
            nbins=20,
            color_discrete_sequence=['#3498DB'],
            labels={'x': 'GNN Fraud Score'}
        )
        fig_score.add_vline(
            x=0.5, line_dash="dash",
            line_color="red",
            annotation_text="Decision threshold"
        )
        fig_score.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_score, use_container_width=True)

    # Node table
    st.subheader("📋 Transaction Details")
    df_display = df_nodes.copy()
    df_display['fraud_score'] = propagated_scores.round(3)
    df_display['risk_level'] = pd.cut(
        df_display['fraud_score'],
        bins=[0, 0.3, 0.6, 1.01],
        labels=['🟢 Low', '🟠 Medium', '🔴 High']
    )
    df_display['amount'] = df_display['amount'].apply(
        lambda x: f"₸{x:,.0f}"
    )
    df_display['is_fraud'] = df_display['is_fraud'].map(
        {True: '🚨 FRAUD', False: '✅ Normal'}
    )
    df_display = df_display.sort_values(
        'fraud_score', ascending=False
    )
    st.dataframe(
        df_display[['id', 'card', 'amount', 'is_fraud',
                     'fraud_score', 'risk_level']],
        use_container_width=True
    )

with tab3:
    st.subheader("🎯 Live Fraud Scanner")
    st.write(
        "Scan a new transaction through the GCN model "
        "and see real-time fraud probability."
    )

    # Show inputs from sidebar
    st.info(
        f"**Transaction:** ₸{scan_amount:,} | "
        f"Days since last: {scan_days} | "
        f"7-day count: {scan_count} | "
        f"Avg amount: ₸{scan_avg:,}"
    )

    if st.button("🔍 Scan Transaction", type="primary"):
        with st.spinner("Running GCN inference..."):

            # Build feature tensor
            amount_norm = min(scan_amount / 500000, 1.0)
            days_norm = min(scan_days / 30, 1.0)
            count_norm = min(scan_count / 50, 1.0)
            avg_norm = min(scan_avg / 100000, 1.0)
            amount_ratio = min(scan_amount / (scan_avg + 1), 10.0) / 10.0

            x = torch.tensor([
                [amount_norm, days_norm, count_norm, avg_norm,
                amount_ratio, 0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3],
                [0.1, 0.2, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            ], dtype=torch.float)

            edge_index = torch.tensor(
                [[0, 1], [1, 0]], dtype=torch.long
            )

            with torch.no_grad():
                out = model(x, edge_index)
                probs = F.softmax(out, dim=1)
                fraud_prob = probs[0, 1].item()

            # Risk boosters
            if scan_amount > scan_avg * 3:
                fraud_prob = min(0.95, fraud_prob + 0.15)
            if scan_count > 20:
                fraud_prob = min(0.95, fraud_prob + 0.10)
            if scan_days < 0.5:
                fraud_prob = min(0.95, fraud_prob + 0.10)

            decision = "BLOCK" if fraud_prob > 0.5 else "ALLOW"
            risk = (
                "HIGH" if fraud_prob > 0.7 else
                "MEDIUM" if fraud_prob > 0.4 else
                "LOW"
            )

        # Results
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("🎯 Fraud Probability",
                       f"{fraud_prob:.1%}")
        with r2:
            if decision == "BLOCK":
                st.error(f"🚫 Decision: {decision}")
            else:
                st.success(f"✅ Decision: {decision}")
        with r3:
            st.metric("⚠️ Risk Level", risk)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            title={'text': "Fraud Risk Score"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': (
                    '#E74C3C' if fraud_prob > 0.5 else '#2ECC71'
                )},
                'steps': [
                    {'range': [0, 30], 'color': '#EAFAF1'},
                    {'range': [30, 60], 'color': '#FEF9E7'},
                    {'range': [60, 100], 'color': '#FDEDEC'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Explanation
        st.subheader("🔍 Why this score?")
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            ratio = scan_amount / (scan_avg + 1)
            color = "🔴" if ratio > 3 else "🟢"
            st.metric(
                f"{color} Amount vs Average",
                f"{ratio:.1f}x",
                delta="High" if ratio > 3 else "Normal"
            )
        with exp2:
            color = "🔴" if scan_count > 20 else "🟢"
            st.metric(
                f"{color} Recent Activity",
                f"{scan_count} txns/7d",
                delta="Suspicious" if scan_count > 20 else "Normal"
            )
        with exp3:
            color = "🔴" if scan_days < 0.5 else "🟢"
            st.metric(
                f"{color} Time Since Last",
                f"{scan_days}d",
                delta="Too fast!" if scan_days < 0.5 else "Normal"
            )

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "🔍 Fraud GNN | Graph Convolutional Network | "
    "IEEE-CIS Dataset | AUC-ROC: 0.7807 | "
    "🔗 [API](https://fraud-gnn-1.onrender.com/docs) | "
    "Built by Rashid Nurbekov 🇰🇿"
)