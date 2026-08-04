"""
shader.py
---------
All GLSL (#version 330 core) shader source, kept as plain Python strings so
renderer.py can compile them with moderngl without needing separate .glsl
asset files.

Five programs are used:

  1. Background program  : draws the webcam frame as a full-screen textured
                            quad with a dark cyan "hologram" tint.
  2. Mesh panel program   : fills Delaunay triangles with a translucent
                            procedural grain/scanline texture.
  3. Line program         : draws subtle mesh edge highlights AND per-hand skeleton
                            bones. Geometry is built on the CPU as thin
                            camera-facing QUADS (two triangles) in pixel
                            space, because glLineWidth > 1 is unreliable on
                            core-profile drivers. The fragment shader uses a
                            per-fragment radial falloff across the quad's
                            width to fake a soft neon GLOW + built-in
                            anti-aliased edges.
  4. Point program        : draws each landmark as a camera-facing glowing
                             dot (quad + radial falloff in the fragment
                             shader), same glow trick as above but radial in
                             both axes.
  5. Overlay program      : draws the HUD (FPS, per-point coordinates,
                             labels) which is pre-rendered each frame onto a
                             CPU RGBA canvas (via OpenCV) and uploaded as a
                             texture, then alpha-blended on top of
                             everything else.
"""

# --------------------------------------------------------------------------- #
# 1. Background (webcam quad)
# --------------------------------------------------------------------------- #
BACKGROUND_VERTEX = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT = """
#version 330 core
uniform sampler2D u_tex;
uniform float u_brightness;
uniform vec3 u_tint;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_tex, v_uv).rgb;
    vec3 tinted = mix(c, c * u_tint, 0.45);
    f_color = vec4(tinted * u_brightness, 1.0);
}
"""

# --------------------------------------------------------------------------- #
# 2. Glowing lines (mesh wireframe + skeleton), CPU-built quads in pixel space
# --------------------------------------------------------------------------- #
LINE_VERTEX = """
#version 330 core
uniform mat4 u_proj;
in vec2 in_pos;      // pixel-space position
in float in_side;    // -1..1 across the width of the line quad
in vec3 in_color;
in float in_alpha;
out float v_side;
out vec3 v_color;
out float v_alpha;
void main() {
    v_side = in_side;
    v_color = in_color;
    v_alpha = in_alpha;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

LINE_FRAGMENT = """
#version 330 core
uniform float u_glow_power;
in float v_side;
in vec3 v_color;
in float v_alpha;
out vec4 f_color;
void main() {
    float d = clamp(abs(v_side), 0.0, 1.0);
    // soft, anti-aliased falloff from the centerline to the quad edge
    float falloff = pow(1.0 - d, u_glow_power);
    f_color = vec4(v_color, v_alpha * falloff);
}
"""

# --------------------------------------------------------------------------- #
# 3. Glowing points (landmarks), CPU-built billboards in pixel space
# --------------------------------------------------------------------------- #
POINT_VERTEX = """
#version 330 core
uniform mat4 u_proj;
in vec2 in_pos;    // pixel-space position of this quad corner
in vec2 in_uv;     // -1..1, -1..1 across the quad
in vec3 in_color;
in float in_alpha;
out vec2 v_uv;
out vec3 v_color;
out float v_alpha;
void main() {
    v_uv = in_uv;
    v_color = in_color;
    v_alpha = in_alpha;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

POINT_FRAGMENT = """
#version 330 core
uniform float u_glow_power;
in vec2 v_uv;
in vec3 v_color;
in float v_alpha;
out vec4 f_color;
void main() {
    float r = length(v_uv);
    if (r > 1.0) discard;
    float core = smoothstep(0.30, 0.0, r);            // bright solid center
    float glow = pow(smoothstep(1.0, 0.0, r), u_glow_power);
    float intensity = max(core, glow * 0.65);
    if (intensity < 0.01) discard;
    f_color = vec4(v_color, v_alpha * intensity);
}
"""

# --------------------------------------------------------------------------- #
# 4. HUD overlay (FPS / coordinates text, pre-rendered to an RGBA texture)
# --------------------------------------------------------------------------- #
OVERLAY_VERTEX = BACKGROUND_VERTEX

OVERLAY_FRAGMENT = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""

# --------------------------------------------------------------------------- #
# 5. Textured translucent mesh panels (filled Delaunay triangles)
# --------------------------------------------------------------------------- #
MESH_PANEL_VERTEX = """
#version 330 core
uniform mat4 u_proj;
in vec2 in_pos;   // pixel-space position
in vec2 in_uv;    // normalized camera-space coordinate
in vec3 in_bary;  // barycentric coordinate for soft triangle borders
out vec2 v_uv;
out vec3 v_bary;
void main() {
    v_uv = in_uv;
    v_bary = in_bary;
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
}
"""

MESH_PANEL_FRAGMENT = """
#version 330 core
uniform vec3 u_color;
uniform float u_alpha;
uniform float u_grain_scale;
uniform float u_edge_alpha;
in vec2 v_uv;
in vec3 v_bary;
out vec4 f_color;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    float grain_a = hash(floor(v_uv * u_grain_scale));
    float grain_b = hash(floor((v_uv + 0.37) * u_grain_scale * 0.53));
    float grain = mix(grain_a, grain_b, 0.35);

    float edge_dist = min(min(v_bary.x, v_bary.y), v_bary.z);
    float edge = 1.0 - smoothstep(0.0, 0.045, edge_dist);
    float scan = 0.5 + 0.5 * sin((v_uv.x + v_uv.y) * 95.0);

    vec3 panel = u_color * (0.58 + grain * 0.32 + scan * 0.10);
    float alpha = u_alpha * (0.72 + grain * 0.35) + edge * u_edge_alpha;
    f_color = vec4(panel, clamp(alpha, 0.0, 1.0));
}
"""
