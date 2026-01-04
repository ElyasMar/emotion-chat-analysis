import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Emotion Chat Analyzer", page_icon="🎭", layout="wide")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

st.title("🎭 Emotion-Aware Chat Analyzer")
st.markdown("Analyze emotions in real-time and get actionable recommendations")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", "http://localhost:5000")
    
    st.header("About")
    st.info("""
    This app analyzes emotions in text using ML models:
    - **Primary Emotion**: Main detected emotion
    - **Confidence**: Model confidence score
    - **Recommendations**: Suggested actions
    """)
    
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.success("History cleared!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Test Your Emotions In The Chat")
    
    user_input = st.text_area("Enter your text:", height=100, key="input_text")
    
    if st.button("Analyze Emotion", type="primary"):
        if user_input:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"text": user_input},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Add to history
                        st.session_state.conversation_history.append({
                            'timestamp': datetime.now(),
                            'text': user_input,
                            'emotion': result['primary_emotion'],
                            'confidence': result['confidence'],
                            'recommendation': result['recommendation']
                        })
                        
                        # Display result
                        st.success(f"**Primary Emotion:** {result['primary_emotion'].upper()}")
                        st.info(f"**Confidence:** {result['confidence']:.2%}")
                        st.warning(f"**Recommendation:** {result['recommendation']}")
                        
                        # Top emotions
                        st.subheader("Top Emotions")
                        for emotion_data in result['top_emotions']:
                            st.progress(emotion_data['confidence'], 
                                       text=f"{emotion_data['emotion']}: {emotion_data['confidence']:.2%}")
                    else:
                        st.error(f"Error: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please enter some text")

with col2:
    st.header("Emotional Trends")
    
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history)
        
        # Emotion distribution
        emotion_counts = df['emotion'].value_counts()
        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                    title="Emotion Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence over time
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence'
        ))
        fig2.update_layout(title="Confidence Trend", xaxis_title="Message #", yaxis_title="Confidence")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No conversation history yet")

# Conversation history
st.header("Conversation History")
if st.session_state.conversation_history:
    for idx, entry in enumerate(reversed(st.session_state.conversation_history[-10:]), 1):
        with st.expander(f"Message {len(st.session_state.conversation_history) - idx + 1} - {entry['emotion']}"):
            st.write(f"**Text:** {entry['text']}")
            st.write(f"**Emotion:** {entry['emotion']} ({entry['confidence']:.2%})")
            st.write(f"**Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Recommendation:** {entry['recommendation']}")
else:
    st.info("Start analyzing text to build conversation history")