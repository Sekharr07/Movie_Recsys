"""Streamlit demo.   streamlit run app.py"""
import pandas as pd
import streamlit as st

from recsys.service import RecService

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Training models (one-time, ~30s)...")
def get_service():
    return RecService()


svc = get_service()
st.title("🎬 Movie Recommender")
st.caption("MovieLens 100K · Hybrid of item-based CF and BPR matrix factorization · "
           "popularity-based cold start")

tab1, tab2, tab3 = st.tabs(["Existing user", "New user (cold start)", "Similar movies"])

with tab1:
    c1, c2, c3 = st.columns([1, 1, 2])
    user = c1.number_input("User id", 0, svc.nu - 1, 0)
    model = c2.selectbox("Model", list(svc.models))
    k = c3.slider("How many?", 5, 20, 10)
    left, right = st.columns(2)
    with left:
        st.subheader("What they liked before")
        st.dataframe(pd.DataFrame(svc.history(int(user))), hide_index=True)
    with right:
        st.subheader("Recommended for them")
        recs = svc.recommend(int(user), k, model)
        st.dataframe(pd.DataFrame([{
            "Movie": r["title"],
            "Because they liked": ", ".join(r["because_you_liked"]) or "-"} for r in recs]),
            hide_index=True)
    st.caption("'Because they liked' comes from item-item similarity, so it is an "
               "explanation of the neighbourhood signal, not of BPR's latent factors.")

with tab2:
    st.write("No history yet? Pick genres and we rank well-rated movies in them.")
    picked = st.multiselect("Favourite genres", [g for g in svc.genres if g != "unknown"],
                            default=["Comedy"])
    st.dataframe(pd.DataFrame(svc.cold_start(picked, 10)), hide_index=True)

with tab3:
    options = {svc.title(i): i for i in range(svc.ni)}
    choice = st.selectbox("Pick a movie", sorted(options))
    st.dataframe(pd.DataFrame(svc.similar(options[choice], 10)), hide_index=True,
                 use_container_width=True)
