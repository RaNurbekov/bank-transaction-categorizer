import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Bank Transaction Categorizer",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Bank Transaction Categorizer")
st.caption(
    "DistilBERT fine-tuned on bank transactions — "
    "classifies dirty POS terminal descriptions into spending categories"
)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "./models/transaction_nlp"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

with st.spinner("Loading DistilBERT weights... 🧠"):
    tokenizer, model = load_model()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("💡 Quick Examples")
st.sidebar.caption("Click to test instantly:")

examples = {
    "🚕 Transport": "YANDEX*TAXI ALMATY KAZ 4512",
    "🛍️ Shopping": "KASPI SHOP PURCHASE 7821",
    "🍕 Food": "GLOVO*ORDER 9921 ALMATY",
    "💡 Utilities": "KCELL MOBILE PAYMENT",
    "🏦 Transfer": "KASPI TRANSFER TO CARD",
    "🏥 Healthcare": "APTEKA 36.6 ALMATY",
    "🎬 Entertainment": "NETFLIX SUBSCRIPTION",
    "✈️ Transport": "AIR ASTANA TICKET 4521",
    "🛒 Shopping": "WILDBERRIES ORDER 55123",
    "⚡ Utilities": "ALMATY ENERGO ELECTRIC",
}

selected_example = None
for label, text in examples.items():
    if st.sidebar.button(label, key=text):
        selected_example = text

st.sidebar.divider()
st.sidebar.subheader("📊 Model Info")
st.sidebar.info("""
**Model:** DistilBERT (multilingual)
**Fine-tuned on:** Bank transactions
**Input:** Dirty POS terminal text
**Output:** Spending category + confidence
""")

# ── Main input ────────────────────────────────────────────
st.subheader("🔍 Classify Transaction")

default_text = selected_example if selected_example else "YANDEX*TAXI ALMATY KAZ 4512"
user_input = st.text_input(
    "Transaction description:",
    value=default_text,
    placeholder="Enter dirty bank transaction text..."
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    predict_btn = st.button(
        "🔍 Classify", type="primary", use_container_width=True
    )
with col2:
    st.write("")
with col3:
    st.write("")

# ── Prediction ────────────────────────────────────────────
if predict_btn or selected_example:
    if user_input.strip() == "":
        st.warning("Please enter a transaction description!")
    else:
        with st.spinner("Running DistilBERT inference..."):
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            )

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                logits, dim=-1
            )[0]

            predicted_class_id = probabilities.argmax().item()
            predicted_label = model.config.id2label[predicted_class_id]
            confidence = probabilities[predicted_class_id].item()

            # Top-5 categories
            top5_probs, top5_ids = torch.topk(probabilities, min(5, len(probabilities)))
            top5_labels = [
                model.config.id2label[idx.item()]
                for idx in top5_ids
            ]
            top5_values = [p.item() for p in top5_probs]

        st.divider()

        # ── Result cards ──────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("🏷️ Category", predicted_label)
        with r2:
            st.metric("🎯 Confidence", f"{confidence:.1%}")
        with r3:
            risk = (
                "✅ High" if confidence > 0.8 else
                "⚠️ Medium" if confidence > 0.5 else
                "❌ Low"
            )
            st.metric("📊 Certainty", risk)

        # ── Confidence banner ─────────────────────────────
        if confidence > 0.8:
            st.success(f"**{predicted_label}** — {confidence:.1%} confidence")
        elif confidence > 0.5:
            st.warning(f"**{predicted_label}** — {confidence:.1%} confidence")
        else:
            st.error(
                f"**{predicted_label}** — Low confidence ({confidence:.1%}). "
                "Transaction text may be too noisy."
            )

        # ── Top-5 bar chart ───────────────────────────────
        st.subheader("📊 Top-5 Categories")
        fig = go.Figure(go.Bar(
            x=top5_values,
            y=top5_labels,
            orientation='h',
            marker=dict(
                color=top5_values,
                colorscale=[
                    [0.0, '#EBF5FB'],
                    [0.5, '#3498DB'],
                    [1.0, '#1A5276']
                ],
            ),
            text=[f"{v:.1%}" for v in top5_values],
            textposition='outside'
        ))
        fig.update_layout(
            xaxis=dict(
                tickformat='.0%',
                range=[0, 1.1]
            ),
            yaxis=dict(autorange="reversed"),
            height=300,
            margin=dict(l=0, r=80, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Transaction analysis ──────────────────────────
        st.subheader("🔬 Transaction Analysis")
        a1, a2 = st.columns(2)

        with a1:
            st.write("**Input text:**")
            st.code(user_input)
            st.write("**Detected patterns:**")
            text_upper = user_input.upper()
            patterns = []
            if any(x in text_upper for x in ['TAXI', 'UBER', 'YANDEX', 'INDRIVE']):
                patterns.append("🚕 Ride-sharing keywords")
            if any(x in text_upper for x in ['KASPI', 'HALYK', 'TRANSFER']):
                patterns.append("🏦 Banking keywords")
            if any(x in text_upper for x in ['GLOVO', 'WOLT', 'FOOD']):
                patterns.append("🍕 Food delivery keywords")
            if any(x in text_upper for x in ['*', '#', 'KAZ', 'ALM']):
                patterns.append("🔢 POS terminal noise detected")
            if patterns:
                for p in patterns:
                    st.write(f"  • {p}")
            else:
                st.write("  • No specific patterns detected")

        with a2:
            st.write("**All categories:**")
            all_labels = list(model.config.id2label.values())
            all_probs = probabilities.tolist()
            for label, prob in sorted(
                zip(all_labels, all_probs),
                key=lambda x: x[1],
                reverse=True
            ):
                bar_filled = int(prob * 20)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                st.write(f"`{bar}` {label}: {prob:.1%}")

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "💳 Bank Transaction Categorizer | "
    "DistilBERT (multilingual) | "
    "Fine-tuned on bank transaction descriptions | "
    "Built by Rashid Nurbekov 🇰🇿"
)