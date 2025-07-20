"""
Voxtral AI Demo Interface

This Gradio demo interface provides users the ability to interact with the Voxtral AI model, a state-of-the-art solution for speech synthesis and natural language processing.

Voxtral has been evaluated and is found to be highly effective for various tasks such as text-to-speech conversion and language understanding.

Hardware Requirements:
- H100/A100 GPUs for the 24B model
- 3090RTX or lower GPUs for the 3B model

This demo is provided by Mohammad Khalooei in collaboration with the Mixteral project.

For further information or assistance, feel free to contact Mohammad Khalooei at mkhalooei@gmail.com.

Version: 1.0
Date: 2025-07-20
"""

import gradio as gr
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch
import os

# ---------- Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
available_models = ["mistralai/Voxtral-Small-24B-2507", "mistralai/Voxtral-Mini-3B-2507"]
model_cache = {}  # {"model_name": (processor, model)}

# ---------- Model Loader ----------
def load_model(model_id):
    if model_id not in model_cache:
        processor = AutoProcessor.from_pretrained(model_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device
        )
        model_cache[model_id] = (processor, model)
    return f"✅ Loaded {model_id}"

# ---------- Inference Functions ----------
def run_conversation(model_id, content):
    processor, model = model_cache[model_id]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}])
    inputs = inputs.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    outputs = model.generate(**inputs, max_new_tokens=500)
    result = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result[0]

def run_multi_turn(model_id, conversation):
    processor, model = model_cache[model_id]
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    outputs = model.generate(**inputs, max_new_tokens=500)
    result = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result[0]

def run_batch(model_id, conversations):
    processor, model = model_cache[model_id]
    inputs = processor.apply_chat_template(conversations)
    inputs = inputs.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    outputs = model.generate(**inputs, max_new_tokens=500)
    result = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return "\n\n---\n\n".join(result)

def run_transcription(model_id, audio):
    processor, model = model_cache[model_id]
    inputs = processor.apply_transcrition_request(language="en", audio=audio, model_id=model_id)
    inputs = inputs.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    outputs = model.generate(**inputs, max_new_tokens=500)
    result = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result[0]

# ---------- Tab Builders ----------
def build_tab_single_instruction():
    with gr.Blocks() as tab:
        gr.Markdown("### 🔊 Multi-Audio + Text Instruction\nUpload two audio files and a text prompt. The model will reason over the combination of all inputs.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        audio1 = gr.Audio(type="filepath", label="Audio 1")
        audio2 = gr.Audio(type="filepath", label="Audio 2")
        prompt = gr.Textbox(label="Text Prompt")
        submit = gr.Button("Run")
        output = gr.Textbox(label="Response")

        load_btn.click(load_model, inputs=model_selector, outputs=status)
        submit.click(
            lambda m, a1, a2, t: run_conversation(m, [
                {"type": "audio", "path": a1},
                {"type": "audio", "path": a2},
                {"type": "text", "text": t}
            ]),
            inputs=[model_selector, audio1, audio2, prompt],
            outputs=output
        )
    return tab

def build_tab_multi_turn():
    with gr.Blocks() as tab:
        gr.Markdown("### 🧠 Multi-Turn Conversation\nSimulate a multi-turn conversation with the model. It keeps track of previous exchanges to give context-aware responses.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        audio1 = gr.Audio(type="filepath", label="Audio 1")
        audio2 = gr.Audio(type="filepath", label="Audio 2")
        text1 = gr.Textbox(label="First prompt")
        audio3 = gr.Audio(type="filepath", label="Follow-up Audio")
        text2 = gr.Textbox(label="Follow-up prompt")
        submit = gr.Button("Run")
        output = gr.Textbox(label="Response")

        load_btn.click(load_model, inputs=model_selector, outputs=status)

        def wrapped_multi_turn(m, a1, a2, t1, a3, t2):
            return run_multi_turn(m, [
                {
                    "role": "user",
                    "content": [{"type": "audio", "path": a1}, {"type": "audio", "path": a2}, {"type": "text", "text": t1}]
                },
                {
                    "role": "assistant",
                    "content": "Dummy placeholder assistant response from earlier turn."
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "path": a3}, {"type": "text", "text": t2}]
                },
            ])

        submit.click(wrapped_multi_turn, inputs=[model_selector, audio1, audio2, text1, audio3, text2], outputs=output)
    return tab

def build_tab_text_only():
    with gr.Blocks() as tab:
        gr.Markdown("### 💬 Text-Only Instruction\nUse Voxtral in a text-only instruction mode, similar to standard LLMs. Great for pure language tasks.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        prompt = gr.Textbox(label="Text Only Prompt")
        submit = gr.Button("Run")
        output = gr.Textbox(label="Response")

        load_btn.click(load_model, inputs=model_selector, outputs=status)
        submit.click(
            lambda m, t: run_conversation(m, [{"type": "text", "text": t}]),
            inputs=[model_selector, prompt],
            outputs=output
        )
    return tab

def build_tab_audio_only():
    with gr.Blocks() as tab:
        gr.Markdown("### 🎧 Audio-Only Instruction\nUpload a single audio file. The model will analyze and describe or infer insights from the audio.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        audio = gr.Audio(type="filepath", label="Audio Input")
        submit = gr.Button("Run")
        output = gr.Textbox(label="Response")

        load_btn.click(load_model, inputs=model_selector, outputs=status)
        submit.click(
            lambda m, a: run_conversation(m, [{"type": "audio", "path": a}]),
            inputs=[model_selector, audio],
            outputs=output
        )
    return tab

def build_tab_batch():
    with gr.Blocks() as tab:
        gr.Markdown("### 📦 Batched Inference\nRun multiple instruction-style conversations in a single batch. Input should be a JSON list of structured chat turns.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        json_input = gr.Textbox(label="Batched Conversations (JSON list)", lines=10, placeholder="Paste list of conversations...")
        submit = gr.Button("Run")
        output = gr.Textbox(label="Responses")

        load_btn.click(load_model, inputs=model_selector, outputs=status)

        def handle_batch(m, js):
            import json
            conversations = json.loads(js)
            return run_batch(m, conversations)

        submit.click(handle_batch, inputs=[model_selector, json_input], outputs=output)
    return tab

def build_tab_transcription():
    with gr.Blocks() as tab:
        gr.Markdown("### ✍️ Audio Transcription\nUpload an audio file to get a plain transcription of its spoken content.")

        model_selector = gr.Dropdown(choices=available_models, label="Model")
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        audio = gr.Audio(type="filepath", label="Audio to Transcribe")
        submit = gr.Button("Transcribe")
        output = gr.Textbox(label="Transcript")

        load_btn.click(load_model, inputs=model_selector, outputs=status)
        submit.click(run_transcription, inputs=[model_selector, audio], outputs=output)
    return tab

# ---------- Launch App ----------
with gr.Blocks(title="Voxtral AI Demo") as demo:
    # gr.Image("logo.png", elem_classes="logo", show_label=False)
    gr.Markdown("# Voxtral AI Demo")
    gr.Markdown("Voxtral(https://mistral.ai/news/voxtral) is a cutting-edge AI model for high-quality speech synthesis and language processing. I have evaluated Voxtral and found it highly useful for various applications. This demo is available for use on H100/A100 GPUs for the 24B model and 3090RTX or lower for the 3B Voxtral model.")
    gr.Markdown("Feel free to reach out to Mohammad Khalooei at mkhalooei@gmail.com for further information or assistance, in collaboration with the Mistral project.")

    with gr.Tabs() as tabs:
        with gr.Tab("Multi-Audio + Text"):
            build_tab_single_instruction()
            
        with gr.Tab("Multi-Turn"):
            build_tab_multi_turn()
            
        with gr.Tab("Text Only"):
            build_tab_text_only()
            
        with gr.Tab("Audio Only"):
            build_tab_audio_only()
            
        with gr.Tab("Batched Inference"):
            build_tab_batch()
            
        with gr.Tab("✍️ Transcription"):
            build_tab_transcription()
            

demo.launch(server_name="0.0.0.0", server_port=8087, show_api=False)