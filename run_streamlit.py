import os
import tempfile
from pathlib import Path

import cv2
import torch
import streamlit as st
import numpy as np
from PIL import Image

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import export_to_video, load_image, load_video

from longcat_video.context_parallel import context_parallel_util
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality"
)

EXAMPLE_PROMPTS = {
    "t2v": "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene.",
    "i2v": "A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere.",
    "vc": "A person rides a motorcycle along a long, straight road that stretches between a body of water and a forested hillside. The rider steadily accelerates, keeping the motorcycle centered between the guardrails, while the scenery passes by on both sides. The video captures the journey from the rider’s perspective, emphasizing the sense of motion and adventure.",
}

MODE_OPTIONS = {
    "t2v": "Texto a video",
    "i2v": "Imagen a video",
    "vc": "Continuación de video",
}

MODE_DESCRIPTIONS = {
    "t2v": "Genera un clip nuevo únicamente desde un prompt.",
    "i2v": "Anima una imagen de referencia con instrucciones de movimiento.",
    "vc": "Extiende un video existente manteniendo contexto visual.",
}


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Page configuration
st.set_page_config(
    page_title="LongCat-Video Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 116, 79, 0.16), transparent 30rem),
                radial-gradient(circle at top right, rgba(66, 153, 225, 0.14), transparent 28rem),
                linear-gradient(180deg, #0f172a 0%, #111827 48%, #0b1120 100%);
            color: #f8fafc;
        }
        .hero-card, .info-card {
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(15, 23, 42, 0.72);
            border-radius: 1.2rem;
            padding: 1.35rem;
            box-shadow: 0 24px 80px rgba(2, 6, 23, 0.28);
        }
        .hero-title {
            font-size: clamp(2rem, 5vw, 4.2rem);
            line-height: 0.95;
            font-weight: 900;
            margin: 0;
            letter-spacing: -0.06em;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1.08rem;
            margin-top: 0.8rem;
            max-width: 62rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid rgba(251, 146, 60, 0.48);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            margin: 0 0.35rem 0.35rem 0;
            color: #fed7aa;
            background: rgba(124, 45, 18, 0.24);
            font-size: 0.9rem;
        }
        .metric-card {
            border-radius: 1rem;
            padding: 1rem;
            background: rgba(30, 41, 59, 0.68);
            border: 1px solid rgba(148, 163, 184, 0.20);
            min-height: 7rem;
        }
        .metric-card strong {
            color: #ffffff;
            font-size: 1.05rem;
        }
        .metric-card span {
            color: #cbd5e1;
            display: block;
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.92);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <section class="hero-card">
            <span class="pill">🎬 LongCat-Video</span>
            <span class="pill">T2V · I2V · VC</span>
            <span class="pill">Streamlit front-end</span>
            <h1 class="hero-title">Studio visual para generar video con IA</h1>
            <p class="hero-subtitle">
                Front-end listo para operar el pipeline de LongCat-Video: carga el modelo,
                selecciona el flujo, ajusta parámetros y descarga el resultado desde una interfaz guiada.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


def render_mode_cards():
    cards = st.columns(3)
    for column, (mode, label) in zip(cards, MODE_OPTIONS.items()):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <strong>{label}</strong>
                    <span>{MODE_DESCRIPTIONS[mode]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return original_fps

@st.cache_resource
def load_model(checkpoint_dir):
    """Load model, use cache to avoid reloading"""    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    with st.spinner('Loading model...'):
        cp_split_hw = context_parallel_util.get_optimal_split(1)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch_dtype)
        text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch_dtype)
        vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch_dtype)
        dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch_dtype)

        pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        pipe.to(device)
        
        cfg_step_lora_path = os.path.join(checkpoint_dir, 'lora/cfg_step_lora.safetensors')
        pipe.dit.load_lora(cfg_step_lora_path, 'cfg_step_lora')

        refinement_lora_path = os.path.join(checkpoint_dir, 'lora/refinement_lora.safetensors')
        pipe.dit.load_lora(refinement_lora_path, 'refinement_lora')
    
    return pipe, device

def main():
    inject_custom_css()
    render_hero()
    render_mode_cards()

    st.sidebar.title("⚙️ Panel de control")
    st.sidebar.caption("Configura el modelo y el flujo de generación antes de iniciar el render.")

    checkpoint_dir = st.sidebar.text_input("Directorio del modelo", "./weights/LongCat-Video")
    checkpoint_path = Path(checkpoint_dir).expanduser()

    with st.sidebar.expander("🧭 Guía rápida", expanded=False):
        st.markdown(
            """
            1. Descarga los pesos en `./weights/LongCat-Video`.
            2. Elige el modo: texto, imagen o continuación.
            3. Completa el prompt y sube archivos si aplica.
            4. Pulsa **Generar video** y descarga el MP4.
            """
        )

    with st.expander("💡 Prompts de ejemplo", expanded=False):
        example_tabs = st.tabs(["Texto a video", "Imagen a video", "Continuación"] )
        for tab, key in zip(example_tabs, ["t2v", "i2v", "vc"]):
            with tab:
                st.write(EXAMPLE_PROMPTS[key])

    if not checkpoint_path.exists():
        st.warning(
            f"No se encontró el directorio `{checkpoint_dir}`. Descarga los pesos o ajusta la ruta en el panel lateral."
        )
        st.stop()

    # Load model
    try:
        pipe, device = load_model(str(checkpoint_path))
        st.success(f"Modelo cargado correctamente. Dispositivo: {device}")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {str(e)}")
        return

    # Sidebar - select generation mode
    mode = st.sidebar.selectbox(
        "Modo de generación",
        options=list(MODE_OPTIONS.keys()),
        format_func=lambda x: MODE_OPTIONS[x]
    )

    use_distill = st.sidebar.checkbox("Activar modo distill (más rápido)", value=False)
    use_refine = st.sidebar.checkbox("Activar super-resolución (baja resolución y refinado)", value=False)

    st.sidebar.subheader("Parámetros de generación")
    
    if mode != "t2v":
        resolution = st.sidebar.selectbox("Resolución", ["480p", "720p"], index=0)
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            height = st.number_input("Alto", min_value=256, max_value=1024, value=480, step=64)
        with col2:
            width = st.number_input("Ancho", min_value=256, max_value=1024, value=832, step=64)
    
    num_frames = 93
    
    if use_distill:
        num_inference_steps = 16  # Distill mode: fixed 16 steps
        guidance_scale = 1.0
    else:
        num_inference_steps = 50  # Normal mode: fixed 50 steps
        guidance_scale = 4.0

    seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=2**32-1, value=42)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Entrada")
        
        # Prompt input
        prompt = st.text_area(
            "Prompt positivo",
            height=100,
            placeholder="Describe el contenido, estilo, cámara y movimiento del video..."
        )
        
        negative_prompt = st.text_area(
            "Prompt negativo",
            value=DEFAULT_NEGATIVE_PROMPT,
            height=80,
            disabled=use_distill
        )
        
        # Show different input controls according to mode
        uploaded_file = None
        if mode == "i2v":
            uploaded_file = st.file_uploader(
                "Subir imagen",
                type=['png', 'jpg', 'jpeg'],
                help="Soporta formatos PNG, JPG y JPEG"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen subida", use_container_width=True)
        
        elif mode == "vc":
            uploaded_file = st.file_uploader(
                "Subir video",
                type=['mp4', 'avi', 'mov'],
                help="Soporta formatos MP4, AVI y MOV"
            )
            if uploaded_file:
                st.video(uploaded_file)
            
            num_cond_frames = 13
        
        # Generate button
        generate_btn = st.button("🎬 Generar video", type="primary", width='stretch')
    
    with col2:
        st.subheader("🎥 Resultado")
        result_placeholder = st.empty()
        
        if generate_btn:
            if not prompt.strip():
                st.error("Escribe un prompt antes de generar.")
                return
            
            if mode != "t2v" and uploaded_file is None:
                st.error(f"Sube un archivo de {'imagen' if mode == 'i2v' else 'video'} antes de generar.")
                return
            
            # Set random seed
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            
            # Generate video according to mode
            with st.spinner('Generando video, espera unos minutos...'):
                if mode == "t2v":
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_t2v(
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            prompt="",
                            stage1_video=stage1_video,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
                
                elif mode == "i2v":
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        image.save(tmp_file.name)
                        input_image = load_image(tmp_file.name)
                    
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_i2v(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        resolution=resolution,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            image=input_image,
                            prompt="",
                            stage1_video=stage1_video,
                            num_cond_frames=1,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
                
                elif mode == "vc":
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        input_video = load_video(tmp_file.name)
                        current_fps = get_fps(tmp_file.name)
                    
                    target_fps = 15
                    stride = max(1, round(current_fps / target_fps))
                    if use_distill:
                        pipe.dit.enable_loras(['cfg_step_lora'])
                    output = pipe.generate_vc(
                        video=input_video[::stride],
                        prompt=prompt,
                        negative_prompt=None if use_distill else negative_prompt,
                        resolution=resolution,
                        num_frames=num_frames,
                        num_cond_frames=num_cond_frames,
                        num_inference_steps=num_inference_steps,
                        use_distill=use_distill,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        use_kv_cache=True,
                        offload_kv_cache=False,
                        enhance_hf=False if use_distill else True
                    )[0]
                    pipe.dit.disable_all_loras()
                    torch_gc()

                    if use_refine:
                        pipe.dit.enable_loras(['refinement_lora'])
                        stage1_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
                        stage1_video = [Image.fromarray(img) for img in stage1_video]
                        del output
                        target_fps = 30
                        stride = max(1, round(current_fps / target_fps))
                        pipe.dit.enable_bsa()
                        output = pipe.generate_refine(
                            video=input_video[::stride],
                            prompt="",
                            stage1_video=stage1_video,
                            num_cond_frames=num_cond_frames*2,
                            num_inference_steps=50,
                            generator=generator
                        )[0]
                        pipe.dit.disable_all_loras()
                        pipe.dit.disable_bsa()
                        torch_gc()
            
            # Save and display result
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_file:
                fps = 30 if use_refine else 15
                export_to_video(output, output_file.name, fps=fps)
                
                with result_placeholder.container():
                    st.success("Generación completada")
                    st.video(output_file.name)
                    
                    # Provide download button
                    with open(output_file.name, 'rb') as f:
                        st.download_button(
                            label="📥 Descargar video",
                            data=f.read(),
                            file_name=f"generated_video_{mode}_{seed}.mp4",
                            mime="video/mp4"
                        )


if __name__ == "__main__":
    main()