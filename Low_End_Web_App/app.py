import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import tempfile
import os
import base64
from models import PredictSegment, PredictSpecies
import plotly.graph_objects as go
import plotly.express as px

# Configure Streamlit page
st.set_page_config(
    page_title="Biofouling Analysis Demo",
    page_icon="🐚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        color: #A23B72;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .species-result {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    # Get the original file extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    if not file_extension:
        file_extension = '.jpg'  # Default fallback
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def create_figure(image_array, title, cmap=None):
    """Create matplotlib figure from numpy array"""
    fig, ax = plt.subplots(figsize=(6, 6))
    if cmap:
        ax.imshow(image_array, cmap=cmap)
    else:
        ax.imshow(image_array)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Convert to bytes for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    plt.close()
    return buf

def create_submarine_heatmap(area_ratio):
    """Create 3D submarine heatmap using Three.js with proper gradients"""
    import json
    
    # Check if submarine.glb exists and encode it as data URL
    glb_path = "submarine.glb"
    if not os.path.exists(glb_path):
        st.error(f"❌ submarine.glb not found at: {os.path.abspath(glb_path)}")
        return None
    
    # Read and encode the GLB file as base64 data URL
    try:
        with open(glb_path, 'rb') as f:
            glb_data = f.read()
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')
        model_data_url = f"data:model/gltf-binary;base64,{glb_base64}"
        st.success(f"✅ Loaded submarine.glb ({len(glb_data)} bytes)")
    except Exception as e:
        st.error(f"❌ Error reading submarine.glb: {e}")
        return None
    
    # Map area ratio to specific submarine segments with gradients
    area_ratio_map = {
        "front_left_top": min(1.0, area_ratio * 2.0),       # Scale up for visibility
        "front_left_bottom": min(1.0, area_ratio * 2.5),    # Higher intensity at bottom
        "middle_left_bottom": min(1.0, area_ratio * 1.8),   # Add more segments
        "front_right_top": min(1.0, area_ratio * 1.2),     # Vary intensities
    }
    
    # Convert to JSON for JavaScript
    area_ratio_json = json.dumps(area_ratio_map)
    
    # Create the Three.js HTML viewer
    html_content = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <style>
            html, body, #three-wrap {{
                margin: 0;
                height: 100%;
                background: #02040a;
            }}
            #three-wrap {{
                width: 100%;
                height: 80vh;
            }}
        </style>
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js",
                "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"
            }}
        }}
        </script>
    </head>
    <body>
        <div id="three-wrap"></div>
        <script type="module">
            import * as THREE from 'three';
            import {{ GLTFLoader }} from 'three/examples/jsm/loaders/GLTFLoader.js';
            import {{ OrbitControls }} from 'three/examples/jsm/controls/OrbitControls.js';
            
            const AREA_RATIO = {area_ratio_json};
            const MODEL_DATA_URL = "{model_data_url}";
            
            // Heat color gradient: 0..1 -> green->yellow->red
            function heatColor01(v) {{
                const t = Math.max(0, Math.min(1, v));
                if (t <= 0.5) {{
                    const k = t / 0.5; // 0..1
                    return new THREE.Color(`rgb(${{Math.round(255*k)}},255,0)`);
                }} else {{
                    const k = (t - 0.5) / 0.5;
                    return new THREE.Color(`rgb(255,${{Math.round(255*(1-k))}},0)`);
                }}
            }}
            
            // Scene setup
            const el = document.getElementById('three-wrap');
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.setSize(el.clientWidth, el.clientHeight);
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            el.appendChild(renderer.domElement);
            
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x02040a);
            
            const camera = new THREE.PerspectiveCamera(50, el.clientWidth / el.clientHeight, 0.1, 200);
            camera.position.set(0, 2.2, 6);
            scene.add(camera);
            
            const ambient = new THREE.AmbientLight(0xffffff, 0.7);
            const dir = new THREE.DirectionalLight(0xffffff, 0.9);
            dir.position.set(5, 10, 7);
            scene.add(ambient, dir);
            
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            
            // Load GLB from embedded data and apply heat gradients
            const loader = new GLTFLoader();
            console.log('Loading embedded submarine model...');
            
            loader.load(
                MODEL_DATA_URL,
                (gltf) => {{
                    console.log('Successfully loaded embedded submarine model');
                    const root = gltf.scene;
                    
                    root.traverse((obj) => {{
                        if (!obj.isMesh) return;
                        
                        console.log('Found mesh:', obj.name);
                        
                        // Check if this object has a defined area ratio
                        const hasValue = Object.prototype.hasOwnProperty.call(AREA_RATIO, obj.name);
                        const value = hasValue ? AREA_RATIO[obj.name] : -1;
                        
                        // Base color: green if no biofouling data, heat color if present
                        const baseColor = (value < 0) ? new THREE.Color('#2ecc71') : heatColor01(value);
                        
                        // Create translucent material with gradient color
                        const material = new THREE.MeshPhysicalMaterial({{
                            color: baseColor,
                            roughness: 0.8,
                            metalness: 0.1,
                            transparent: true,
                            opacity: hasValue ? 0.75 : 0.55,  // More opaque for biofouling areas
                            transmission: 0.0,
                            depthWrite: true,
                        }});
                        
                        // Subtle wireframe overlay for structure
                        const wireframeMaterial = new THREE.MeshStandardMaterial({{
                            color: new THREE.Color(0x11c5d9),
                            wireframe: true,
                            transparent: true,
                            opacity: 0.2,
                        }});
                        
                        // Create base mesh with gradient color
                        const baseMesh = new THREE.Mesh(obj.geometry, material);
                        baseMesh.position.copy(obj.position);
                        baseMesh.quaternion.copy(obj.quaternion);
                        baseMesh.scale.copy(obj.scale);
                        baseMesh.renderOrder = 0;
                        
                        // Create wireframe overlay
                        const wireframeMesh = new THREE.Mesh(obj.geometry, wireframeMaterial);
                        wireframeMesh.position.copy(obj.position);
                        wireframeMesh.quaternion.copy(obj.quaternion);
                        wireframeMesh.scale.copy(obj.scale);
                        wireframeMesh.renderOrder = 1;
                        
                        // Hide original and add new meshes
                        obj.visible = false;
                        obj.parent.add(baseMesh);
                        obj.parent.add(wireframeMesh);
                    }});
                    
                    scene.add(root);
                    fitView(root);
                }},
                undefined,
                (err) => {{
                    console.error('Failed to load embedded submarine model:', err);
                    el.innerHTML = '<p style="color:white; padding:20px;">Failed to load submarine.glb model. Error: ' + err.message + '</p>';
                }}
            );
            
            function fitView(object3D) {{
                const box = new THREE.Box3().setFromObject(object3D);
                const size = new THREE.Vector3();
                const center = new THREE.Vector3();
                box.getSize(size);
                box.getCenter(center);
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / (2 * Math.tan(fov / 2)));
                cameraZ *= 1.4;
                camera.position.set(center.x, center.y + maxDim * 0.15, center.z + cameraZ);
                camera.lookAt(center);
                controls.target.copy(center);
                controls.update();
            }}
            
            window.addEventListener('resize', () => {{
                const w = el.clientWidth, h = el.clientHeight;
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
                renderer.setSize(w, h);
            }});
            
            (function animate() {{
                controls.update();
                renderer.render(scene, camera);
                requestAnimationFrame(animate);
            }})();
        </script>
    </body>
    </html>
    """
    
    return html_content

def main():
    # Header
    st.markdown('<h1 class="main-header">🐚 Biofouling Analysis Demo</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Upload an image to analyze biofouling segmentation and species classification
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for mode selection and file upload
    with st.sidebar:
        st.markdown("### 🎯 Analysis Mode")
        analysis_mode = st.selectbox(
            "Choose analysis type:",
            ["Segmentation", "Classification"],
            help="Select whether to perform segmentation or species classification"
        )
        
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help=f"Upload an image for {analysis_mode.lower()} analysis"
        )
        
        st.markdown("### ℹ️ About")
        if analysis_mode == "Segmentation":
            st.info("""
            **Segmentation Mode:**
            - Detects biofouling regions in images
            - Outputs mask overlay, outline, and raw mask
            - Calculates area ratio metrics
            - Uses U-Net architecture
            """)
        else:
            st.info("""
            **Classification Mode:**
            - Identifies biofouling species types
            - Recognizes 11 different categories
            - Provides species descriptions
            - Uses Multi-label CNN
            """)
    
    if uploaded_file is not None:
        # Display original image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h2 class="sub-header">📸 Original Image</h2>', unsafe_allow_html=True)
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process the image based on selected mode
        with st.spinner(f"🔄 Processing image for {analysis_mode.lower()}..."):
            try:
                # Save uploaded file temporarily
                temp_path = save_uploaded_file(uploaded_file)
                
                if analysis_mode == "Segmentation":
                    # Run segmentation only
                    overlay, outline, mask, area_ratio = PredictSegment(temp_path)
                    species = None
                else:
                    # Run classification only
                    species = PredictSpecies(temp_path)
                    overlay, outline, mask, area_ratio = None, None, None, None
                
                # Clean up temp file
                os.unlink(temp_path)
                
                st.success(f"✅ {analysis_mode} analysis complete!")
                
                # Display results based on analysis mode
                if analysis_mode == "Segmentation":
                    # Segmentation results
                    st.markdown('<h2 class="sub-header">🎯 Segmentation Results</h2>', unsafe_allow_html=True)
                    
                    # Segmentation results in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Mask Overlay (a)**")
                        overlay_fig = create_figure(overlay, "Mask Overlay")
                        st.image(overlay_fig, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Mask Outline (b)**")
                        outline_fig = create_figure(outline, "Mask Outline")
                        st.image(outline_fig, use_column_width=True)
                    
                    with col3:
                        st.markdown("**Raw Mask (c)**")
                        mask_fig = create_figure(mask, "Raw Mask", cmap='viridis')
                        st.image(mask_fig, use_column_width=True)
                    
                    # Segmentation metrics
                    st.markdown('<h2 class="sub-header">📈 Segmentation Metrics</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Area Ratio (d)")
                        st.markdown(f'''
                        <div class="metric-card">
                            <h3 style="color: #2E86AB; margin: 0;">{area_ratio:.4f}</h3>
                            <p style="margin: 0.5rem 0 0 0; color: #666;">
                                Ratio of biofouling area to total image area
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Additional metrics
                        percentage = area_ratio * 100
                        if percentage < 10:
                            status = "Low Coverage"
                            color = "#28a745"
                        elif percentage < 30:
                            status = "Moderate Coverage"
                            color = "#ffc107"
                        else:
                            status = "High Coverage"
                            color = "#dc3545"
                        
                        st.markdown(f'''
                        <div style="background-color: {color}20; padding: 0.5rem; border-radius: 0.5rem; border-left: 4px solid {color}; margin-top: 1rem;">
                            <strong>{status}</strong><br>
                            {percentage:.2f}% of image area
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### Segmentation Summary")
                        st.markdown(f'''
                        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2E86AB; color: black;">
                            <strong>🔍 Detected Regions:</strong><br>
                            • Total pixels analyzed: {mask.shape[0] * mask.shape[1]:,}<br>
                            • Biofouling pixels: {int((mask > 0).sum()):,}<br>
                            • Clean pixels: {int((mask == 0).sum()):,}<br>
                            • Coverage: {percentage:.2f}%
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Mask statistics
                        unique_values = np.unique(mask)
                        st.markdown("**Mask Classes:**")
                        for i, val in enumerate(unique_values):
                            count = (mask == val).sum()
                            percent = (count / mask.size) * 100
                            st.markdown(f"- Class {int(val)}: {count:,} pixels ({percent:.1f}%)")
                    
                    # 3D Submarine Heatmap Visualization
                    st.markdown('<h2 class="sub-header">🚢 3D Submarine Heatmap</h2>', unsafe_allow_html=True)
                    
                    with st.spinner("🔄 Generating submarine heatmap visualization..."):
                        submarine_html = create_submarine_heatmap(area_ratio)
                        
                        if submarine_html:
                            # Display the Three.js submarine heatmap
                            st.components.v1.html(submarine_html, width=900, height=650, scrolling=False)
                            
                            st.markdown(f'''
                            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; color: black;">
                                <strong>🎯 3D Submarine Visualization:</strong><br>
                                • <strong>Technology</strong>: Three.js WebGL renderer with proper gradient support<br>
                                • <strong>Target segments</strong>: front_left_top, front_left_bottom, middle_left_bottom, front_right_top<br>
                                • <strong>Heat gradient</strong>: Green → Yellow → Red based on area ratio {area_ratio:.4f}<br>
                                • <strong>Default color</strong>: Unspecified segments show in green (clean)<br>
                                • <strong>Interactive controls</strong>: Mouse drag to rotate, scroll to zoom, full 3D navigation<br>
                                • <strong>Natural proportions</strong>: Submarine maintains authentic shape and scale
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.warning("⚠️ Failed to generate 3D submarine visualization")
                    
                    # Comprehensive Data Analysis Section
                    st.markdown('<h2 class="sub-header">📈 Comprehensive Data Analysis</h2>', unsafe_allow_html=True)
                    
                    # Create analysis plots in multiple rows
                    
                    # Row 1: Pixel distribution and coverage analysis
                    st.markdown("### 📊 Pixel Distribution Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of mask classes
                        unique_values, counts = np.unique(mask, return_counts=True)
                        class_names = [f'Class {int(val)}' if val != 0 else 'Background' for val in unique_values]
                        
                        fig_pie = go.Figure(data=[
                            go.Pie(
                                labels=class_names,
                                values=counts,
                                hole=0.4,
                                marker=dict(
                                    colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'],
                                    line=dict(color='#000000', width=2)
                                )
                            )
                        ])
                        fig_pie.update_layout(
                            title="Mask Class Distribution",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Bar chart of pixel counts
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=class_names,
                                y=counts,
                                marker_color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'][:len(class_names)],
                                text=[f'{count:,}' for count in counts],
                                textposition='auto'
                            )
                        ])
                        fig_bar.update_layout(
                            title="Pixel Count by Class",
                            xaxis_title="Mask Classes",
                            yaxis_title="Pixel Count",
                            height=400
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Row 2: Spatial distribution analysis
                    st.markdown("### 🗺️ Spatial Distribution Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Horizontal distribution (row-wise)
                        row_sums = (mask > 0).sum(axis=1)  # Biofouling pixels per row
                        fig_row = go.Figure()
                        fig_row.add_trace(go.Scatter(
                            x=list(range(len(row_sums))),
                            y=row_sums,
                            mode='lines+markers',
                            name='Biofouling Pixels',
                            line=dict(color='#EF553B', width=2),
                            marker=dict(size=4)
                        ))
                        fig_row.update_layout(
                            title="Horizontal Distribution (Top to Bottom)",
                            xaxis_title="Row Index",
                            yaxis_title="Biofouling Pixels",
                            height=350
                        )
                        st.plotly_chart(fig_row, use_container_width=True)
                    
                    with col2:
                        # Vertical distribution (column-wise)
                        col_sums = (mask > 0).sum(axis=0)  # Biofouling pixels per column
                        fig_col = go.Figure()
                        fig_col.add_trace(go.Scatter(
                            x=list(range(len(col_sums))),
                            y=col_sums,
                            mode='lines+markers',
                            name='Biofouling Pixels',
                            line=dict(color='#00CC96', width=2),
                            marker=dict(size=4)
                        ))
                        fig_col.update_layout(
                            title="Vertical Distribution (Left to Right)",
                            xaxis_title="Column Index",
                            yaxis_title="Biofouling Pixels",
                            height=350
                        )
                        st.plotly_chart(fig_col, use_container_width=True)
                    
                    # Row 3: Intensity and density analysis
                    st.markdown("### 🔥 Intensity & Density Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Mask value histogram
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=mask.flatten(),
                            nbinsx=20,
                            marker_color='#AB63FA',
                            opacity=0.7
                        ))
                        fig_hist.update_layout(
                            title="Mask Value Distribution",
                            xaxis_title="Mask Value",
                            yaxis_title="Frequency",
                            height=350
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    with col2:
                        # Coverage heatmap (downsampled for performance)
                        mask_small = cv2.resize(mask.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=mask_small,
                            colorscale='Viridis',
                            showscale=True
                        ))
                        fig_heatmap.update_layout(
                            title="Coverage Heatmap (Downsampled)",
                            height=350,
                            xaxis_title="X Position",
                            yaxis_title="Y Position"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Row 4: Statistical analysis
                    st.markdown("### 📊 Statistical Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Area ratio over different regions (quadrants)
                        h, w = mask.shape
                        quadrants = {
                            'Top-Left': mask[:h//2, :w//2],
                            'Top-Right': mask[:h//2, w//2:],
                            'Bottom-Left': mask[h//2:, :w//2],
                            'Bottom-Right': mask[h//2:, w//2:]
                        }
                        
                        quad_ratios = []
                        quad_names = []
                        for name, quad in quadrants.items():
                            ratio = (quad > 0).sum() / quad.size
                            quad_ratios.append(ratio)
                            quad_names.append(name)
                        
                        fig_quad = go.Figure(data=[
                            go.Bar(
                                x=quad_names,
                                y=quad_ratios,
                                marker_color=['#FFA15A', '#19D3F3', '#FF6692', '#B6E880'],
                                text=[f'{ratio:.3f}' for ratio in quad_ratios],
                                textposition='auto'
                            )
                        ])
                        fig_quad.update_layout(
                            title="Coverage by Quadrants",
                            xaxis_title="Image Quadrant",
                            yaxis_title="Coverage Ratio",
                            height=350
                        )
                        st.plotly_chart(fig_quad, use_container_width=True)
                    
                    with col2:
                        # Biofouling density vs position analysis
                        center_x, center_y = w // 2, h // 2
                        y_coords, x_coords = np.ogrid[:h, :w]
                        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                        
                        # Create distance bins
                        max_dist = np.sqrt(center_x**2 + center_y**2)
                        bins = np.linspace(0, max_dist, 10)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        
                        coverage_by_distance = []
                        for i in range(len(bins)-1):
                            mask_bin = (distances >= bins[i]) & (distances < bins[i+1])
                            if mask_bin.sum() > 0:
                                coverage = (mask[mask_bin] > 0).mean()
                                coverage_by_distance.append(coverage)
                            else:
                                coverage_by_distance.append(0)
                        
                        fig_radial = go.Figure()
                        fig_radial.add_trace(go.Scatter(
                            x=bin_centers,
                            y=coverage_by_distance,
                            mode='lines+markers',
                            name='Coverage',
                            line=dict(color='#FF6692', width=3),
                            marker=dict(size=6)
                        ))
                        fig_radial.update_layout(
                            title="Coverage vs Distance from Center",
                            xaxis_title="Distance from Center (pixels)",
                            yaxis_title="Coverage Ratio",
                            height=350
                        )
                        st.plotly_chart(fig_radial, use_container_width=True)
                    
                    # Row 5: Advanced metrics
                    st.markdown("### 🏅 Advanced Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    # Calculate advanced metrics
                    total_pixels = mask.size
                    biofouling_pixels = (mask > 0).sum()
                    clean_pixels = (mask == 0).sum()
                    
                    # Connectivity analysis (number of connected components)
                    binary_mask = (mask > 0).astype(np.uint8)
                    num_labels, labels = cv2.connectedComponents(binary_mask)
                    num_components = num_labels - 1  # Subtract background
                    
                    # Calculate largest component size
                    if num_components > 0:
                        component_sizes = [(labels == i).sum() for i in range(1, num_labels)]
                        largest_component = max(component_sizes)
                        avg_component_size = np.mean(component_sizes)
                    else:
                        largest_component = 0
                        avg_component_size = 0
                    
                    # Fragmentation index
                    fragmentation = num_components / max(1, biofouling_pixels / 1000)  # Components per 1000 biofouling pixels
                    
                    with col1:
                        st.markdown(f'''
                        <div style="background-color: #e8f4fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; color: black;">
                            <h4 style="color: #1f77b4; margin: 0;">Connectivity Metrics</h4>
                            <p><strong>Connected Components:</strong> {num_components}</p>
                            <p><strong>Largest Component:</strong> {largest_component:,} pixels</p>
                            <p><strong>Average Component:</strong> {avg_component_size:.1f} pixels</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f'''
                        <div style="background-color: #fff2e8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff7f0e; color: black;">
                            <h4 style="color: #ff7f0e; margin: 0;">Fragmentation Analysis</h4>
                            <p><strong>Fragmentation Index:</strong> {fragmentation:.2f}</p>
                            <p><strong>Coverage Uniformity:</strong> {1/max(1, fragmentation):.2f}</p>
                            <p><strong>Distribution Type:</strong> {'Scattered' if fragmentation > 0.1 else 'Clustered'}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'''
                        <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2ca02c; color: black;">
                            <h4 style="color: #2ca02c; margin: 0;">Efficiency Metrics</h4>
                            <p><strong>Pixels Analyzed:</strong> {total_pixels:,}</p>
                            <p><strong>Processing Efficiency:</strong> {(biofouling_pixels/total_pixels)*100:.1f}%</p>
                            <p><strong>Data Density:</strong> {biofouling_pixels/1000:.1f}K biofouling pixels</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                else:
                    # Classification results
                    st.markdown('<h2 class="sub-header">🔬 Classification Results</h2>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown(f'''
                        <div class="species-result">
                            🔬 Detected Species:<br>
                            <span style="color: #2E86AB; font-size: 1.8rem;">{species}</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Species information
                        species_info = {
                            "Algae": "🌿 Marine plant organisms",
                            "Barnacles": "🦦 Small marine crustaceans",
                            "Clean": "✨ No significant biofouling detected",
                            "Hydrozoan": "🪼 Colonial marine organisms",
                            "Jellyfish": "🎐 Gelatinous marine animals",
                            "Mussels": "🦦 Bivalve mollusks",
                            "Rust": "🟤 Corrosion on metal surfaces",
                            "Starfish": "⭐ Echinoderms",
                            "Worms": "🪱 Marine worm species",
                            "Zebra Mussels": "🦓 Invasive freshwater mussels",
                            "Tunicates": "🫧 Marine filter feeders"
                        }
                        
                        if species in species_info:
                            st.markdown(f'''
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem; color: black;">
                                <strong>About this species:</strong><br>
                                <span style="font-size: 1.1rem; color: #333;">{species_info[species]}</span>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        # All species categories
                        st.markdown('<h3 class="sub-header">All Recognized Categories</h3>', unsafe_allow_html=True)
                        
                        # Display all categories in a nice grid
                        categories = list(species_info.keys())
                        for i in range(0, len(categories), 2):
                            cols = st.columns(2)
                            for j, col in enumerate(cols):
                                if i + j < len(categories):
                                    category = categories[i + j]
                                    is_detected = category == species
                                    if is_detected:
                                        style = "background-color: #e8f5e8; border-left: 4px solid #28a745; color: black;"
                                    else:
                                        style = "background-color: #f8f9fa; border-left: 4px solid #ddd; color: black;"
                                    col.markdown(f'''
                                    <div style="{style} padding: 0.5rem; border-radius: 0.3rem; margin: 0.2rem 0;">
                                        <strong style="color: #2E86AB;">{category}</strong><br>
                                        <small style="color: #555;">{species_info[category]}</small>
                                    </div>
                                    ''', unsafe_allow_html=True)
                
                
                # Technical details (collapsible)
                with st.expander("🔧 Technical Details"):
                    if analysis_mode == "Segmentation":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Segmentation Model:**")
                            st.markdown("- Architecture: U-Net with skip connections")
                            st.markdown("- Input: 256x256 RGB images")
                            st.markdown("- Output: 4-class segmentation mask")
                            st.markdown("- Framework: PyTorch")
                            st.markdown("- Preprocessing: Normalized RGB")
                        
                        with col2:
                            st.markdown("**Output Explanation:**")
                            st.markdown("- **a (Overlay)**: Mask overlaid on original image")
                            st.markdown("- **b (Outline)**: Boundary detection of biofouling")
                            st.markdown("- **c (Raw Mask)**: Direct segmentation output")
                            st.markdown("- **d (Area Ratio)**: Biofouling coverage percentage")
                            st.markdown("- **3D Heatmap**: Submarine visualization with gradient")
                            st.markdown("- **Target Segments**: front_left_top, front_left_bottom")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Classification Model:**")
                            st.markdown("- Architecture: Multi-label CNN")
                            st.markdown("- Input: 256x256 RGB images")
                            st.markdown("- Classes: 11 species categories")
                            st.markdown("- Framework: PyTorch")
                            st.markdown("- Preprocessing: Normalized RGB")
                        
                        with col2:
                            st.markdown("**Species Categories:**")
                            st.markdown("- Marine organisms (Algae, Mussels, etc.)")
                            st.markdown("- Colonial species (Hydrozoan, Tunicates)")
                            st.markdown("- Invasive species (Zebra Mussels)")
                            st.markdown("- Corrosion (Rust on metal surfaces)")
                            st.markdown("- Clean surfaces (No biofouling)")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.markdown("Please try with a different image or check that the model files are available.")
    
    else:
        # Instructions when no file is uploaded
        st.markdown("### 👆 Please upload an image to begin analysis")
        
        st.markdown("""
        #### How it works:
        
        1. **Select Analysis Mode** in the sidebar:
           - **Segmentation**: Detect and outline biofouling regions
           - **Classification**: Identify biofouling species types
        
        2. **Upload an image** using the file uploader
        
        3. **View Results**:
           - **Segmentation Mode**: Mask overlay, outline, raw mask, and area metrics
           - **Classification Mode**: Species identification and category information
        
        #### Supported formats:
        - PNG, JPG, JPEG images
        - Recommended: Clear underwater/marine images with visible biofouling
        
        #### Use Cases:
        - **Segmentation**: Quantify biofouling coverage and distribution
        - **Classification**: Identify specific species for targeted treatment
        """)

if __name__ == "__main__":
    main()