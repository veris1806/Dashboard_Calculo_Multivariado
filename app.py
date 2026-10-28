import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración de la pantalla
st.set_page_config(page_title="Monitor de SLA & Servidores", page_icon="🖥️", layout="wide")

# Modelo matemático T(x, y, z) y Gradiente
def latencia(x, y, z):
    return 0.02*x**2 + 0.03*y**2 + 0.01*z**2 - 0.01*x*y - 0.005*x*z - 0.008*y*z + 5

def gradiente(x, y, z):
    dT_dx = 0.04*x - 0.01*y - 0.005*z
    dT_dy = -0.01*x + 0.06*y - 0.008*z
    dT_dz = -0.005*x - 0.008*y + 0.02*z
    return np.array([dT_dx, dT_dy, dT_dz])

# --- MENÚ LATERAL INTERACTIVO ---
st.sidebar.header("⚙️ Muestra / Datos del Servidor")
st.sidebar.markdown("Ingrese o mueva los valores para evaluar el servidor:")

x_in = st.sidebar.number_input("Usuarios Concurrentes (x):", min_value=0, max_value=500, value=100, step=5)
y_in = st.sidebar.number_input("Procesos Activos (y):", min_value=0, max_value=300, value=50, step=5)
z_in = st.sidebar.slider("Consumo de CPU % (z):", min_value=0, max_value=100, value=80, step=1)

st.sidebar.markdown("---")
st.sidebar.header("⚡ Simular Perturbación (Δ Carga)")
dx_in = st.sidebar.number_input("Variación de Usuarios (dx):", value=5)
dy_in = st.sidebar.number_input("Variación de Procesos (dy):", value=-3)
dz_in = st.sidebar.number_input("Variación de CPU % (dz):", value=2)

# --- CÁLCULOS MATEMÁTICOS ---
T_base = latencia(x_in, y_in, z_in)
grad = gradiente(x_in, y_in, z_in)
dT = np.dot(grad, [dx_in, dy_in, dz_in])
T_proyectado = T_base + dT

# --- CUERPO DEL DASHBOARD ---
st.title("🖥️ Dashboard de Monitoreo Predictivo de Latencia y SLA")
st.markdown("Plataforma interactiva para la toma de decisiones en infraestructura mediante cálculo multivariado.")

# Tarjetas de Métricas Visuales
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latencia Base Evaluada", f"{T_base:.2f} ms")
c2.metric("Diferencial Predictivo (dT)", f"{dT:+.2f} ms")
c3.metric("Latencia Proyectada", f"{T_proyectado:.2f} ms", delta=f"{dT:+.2f} ms", delta_color="normal" if T_proyectado < 300 else "inverse")

if T_proyectado >= 300:
    c4.error("🔴 ESTADO CRÍTICO\nViolación de SLA (>= 300 ms)")
elif T_proyectado >= 220:
    c4.warning("🟡 ALERTA PREVENTIVA\nCerca del límite de SLA")
else:
    c4.success("🟢 OPERACIÓN ÓPTIMA\nDentro de la norma")

st.markdown("---")

# Gráfica 3D Interactiva
col_l, col_r = st.columns([2, 1])

with col_l:
    st.subheader("🌐 Superficie de Respuesta 3D (Rotable y con Zoom)")
    X = np.linspace(0, 200, 30)
    Y = np.linspace(0, 100, 30)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = latencia(X_grid, Y_grid, z_in)

    fig = go.Figure(data=[go.Surface(z=Z_grid, x=X_grid, y=Y_grid, colorscale='Viridis', opacity=0.8)])
    fig.add_trace(go.Scatter3d(
        x=[x_in], y=[y_in], z=[T_base],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=["Muestra Evaluada"],
        name="Muestra"
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=450)
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("📋 Análisis del Gradiente")
    st.latex(rf"\nabla T = ({grad[0]:.2f}, {grad[1]:.2f}, {grad[2]:.2f})")
    st.markdown(f"**Magnitud del Gradiente:** `{np.linalg.norm(grad):.3f} ms/unidad`")
    st.info("Escriba valores diferentes en el menú de la izquierda para actualizar las métricas y la gráfica en tiempo real.")
