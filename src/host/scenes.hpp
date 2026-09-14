#pragma once

#include "../core/camera.hpp"
#include "../core/hash_rng.hpp"
#include "../core/scene_view.hpp"
#include "scene_builder.hpp"

#include <string>
#include <vector>

// The ten scenes, built on the host.
//
// Compare these to the old `create_world_*` kernels: same geometry, same
// camera, but ordinary C++ that can be stepped through in a debugger, and no
// `new` anywhere. The result is flat arrays and a BVH, ready to upload.

namespace rt {

enum SceneId {
    SCENE_BOUNCING = 0, SCENE_CHECKERED, SCENE_EARTH, SCENE_PERLIN, SCENE_QUADS,
    SCENE_SIMPLE_LIGHT, SCENE_CORNELL, SCENE_CORNELL_SMOKE, SCENE_FINAL,
    SCENE_ORIGINAL, SCENE_COUNT
};

struct SceneSetup {
    SceneBuilder world;
    CameraSpec   cam;
    vec3         background  = vec3(0, 0, 0);
    bool         gradient_bg = false;
    // Asset paths in the order the scene's TEX_IMAGE descriptors index them.
    std::vector<std::string> images;
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// UT palette, from the original bouncing-spheres scene.
inline vec3 pick_ut_color(float r) {
    if (r < 0.25f) return vec3(1.0f, 1.0f, 1.0f);
    if (r < 0.50f) return vec3(1.0f, 0.51f, 0.0f);   // #FF8200
    if (r < 0.75f) return vec3(0.60f, 0.60f, 0.60f);
    return vec3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------

inline void scene_bouncing(SceneSetup& s, unsigned int seed) {
    auto& w = s.world;
    auto& M = w.materials();

    const vec3 UT_ORANGE(1.0f, 0.51f, 0.0f);
    const int checker = M.checker(0.64f, vec3(1, 1, 1), UT_ORANGE);
    w.add_sphere(vec3(0.0f, -1000.0f, -1.0f), 1000.0f, M.lambertian(checker));

    // The device version drew these from cuRAND inside the build kernel; on the
    // host we use the same deterministic xorshift the tests use, so the scene
    // is reproducible from `seed` without a GPU.
    XorShiftRng rng(seed);
    const float P_EMISSIVE = 0.10f;
    const float EMIT_POWER = 4.0f;

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const float choose_mat = rng.next();
            const vec3 center(a + 0.9f * rng.next(), 0.2f, b + 0.9f * rng.next());

            if (choose_mat < 0.8f) {
                // Diffuse, and moving.
                const vec3 vel(0.0f, 0.5f * rng.next(), 0.25f * (rng.next() - 0.5f));
                if (rng.next() < P_EMISSIVE) {
                    w.add_sphere(center, center + vel, 0.2f,
                                 M.diffuse_light(EMIT_POWER * UT_ORANGE));
                } else {
                    w.add_sphere(center, center + vel, 0.2f,
                                 M.lambertian(pick_ut_color(rng.next())));
                }
            } else if (choose_mat < 0.95f) {
                vec3 albedo = pick_ut_color(rng.next());
                // Pure black metal reads as a void; nudge it to dark grey.
                if (albedo.x() + albedo.y() + albedo.z() < 1e-5f)
                    albedo = vec3(0.15f, 0.15f, 0.15f);
                w.add_sphere(center, 0.2f, M.metal(albedo, 0.5f * rng.next()));
            } else {
                w.add_sphere(center, 0.2f, M.dielectric(1.5f));
            }
        }
    }

    w.add_sphere(vec3( 0.0f, 1.0f, 0.0f), 1.0f, M.dielectric(1.5f));
    w.add_sphere(vec3(-4.0f, 1.0f, 0.0f), 1.0f, M.lambertian(vec3(0.4f, 0.2f, 0.1f)));
    w.add_sphere(vec3( 4.0f, 1.0f, 0.0f), 1.0f, M.metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

    s.cam.lookfrom = vec3(13, 2, 3);
    s.cam.lookat   = vec3(0, 0, 0);
    s.cam.vfov     = 30.0f;
    s.cam.aperture = 0.1f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
    s.cam.time0 = 0.0f; s.cam.time1 = 1.0f;
}

inline void scene_checkered(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    // One material shared by both spheres. Under the old ownership model this
    // was a double free; now materials live in a table and nothing else owns
    // them, so sharing is simply an index used twice.
    const int lam = M.lambertian(M.checker(0.32f, vec3(0.2f, 0.3f, 0.1f), vec3(0.9f, 0.9f, 0.9f)));
    w.add_sphere(vec3(0, -10, 0), 10.0f, lam);
    w.add_sphere(vec3(0,  10, 0), 10.0f, lam);

    s.cam.lookfrom = vec3(13, 2, 3);
    s.cam.lookat   = vec3(0, 0, 0);
    s.cam.vfov     = 20.0f;
    s.cam.focus_dist = 10.0f;
    s.gradient_bg = true;
}

inline void scene_earth(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    s.images.push_back("textures/earthmap.jpg");
    w.add_sphere(vec3(0, 0, 0), 2.0f, M.lambertian(M.image(0)));

    s.cam.lookfrom = vec3(0, 0, 12);
    s.cam.lookat   = vec3(0, 0, 0);
    s.cam.vfov     = 20.0f;
    s.cam.focus_dist = 12.0f;
    s.gradient_bg = true;
}

inline void scene_perlin(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    const int lam = M.lambertian(M.noise(4.0f));
    w.add_sphere(vec3(0, -1000, 0), 1000.0f, lam);
    w.add_sphere(vec3(0,     2, 0),    2.0f, lam);

    s.cam.lookfrom = vec3(13, 2, 3);
    s.cam.lookat   = vec3(0, 0, 0);
    s.cam.vfov     = 20.0f;
    s.cam.focus_dist = 10.0f;
    s.gradient_bg = true;
}

inline void scene_quads(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    w.add_quad(vec3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), M.lambertian(vec3(1.0f, 0.2f, 0.2f)));
    w.add_quad(vec3(-2, -2, 0), vec3(4, 0,  0), vec3(0, 4, 0), M.lambertian(vec3(0.2f, 1.0f, 0.2f)));
    w.add_quad(vec3( 3, -2, 1), vec3(0, 0,  4), vec3(0, 4, 0), M.lambertian(vec3(0.2f, 0.2f, 1.0f)));
    w.add_quad(vec3(-2,  3, 1), vec3(4, 0,  0), vec3(0, 0, 4), M.lambertian(vec3(1.0f, 0.5f, 0.0f)));
    w.add_quad(vec3(-2, -3, 5), vec3(4, 0,  0), vec3(0, 0,-4), M.lambertian(vec3(0.2f, 0.8f, 0.8f)));

    s.cam.lookfrom = vec3(0, 0, 9);
    s.cam.lookat   = vec3(0, 0, 0);
    s.cam.vfov     = 80.0f;
    s.cam.focus_dist = 10.0f;
    s.gradient_bg = true;
}

inline void scene_simple_light(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    s.images.push_back("textures/poolball.jpg");

    const int felt = M.lambertian(M.felt(vec3(0.06f, 0.36f, 0.18f), 16.0f, 0.08f, 4.0f, 0.03f));
    w.add_sphere(vec3(0, -1000, 0), 1000.0f, felt);

    const vec3  C(0, 2, 0);
    const float R = 2.0f;
    // Rotate the decal ~60 degrees toward the camera.
    const int ball = M.lambertian(M.uv_offset(M.image(0), 60.0f / 360.0f));
    w.add_sphere(C, R, ball);
    w.add_sphere(C, R + 0.02f, M.dielectric(1.5f));       // clear-coat lacquer

    w.add_sphere(vec3(0, 7, 0), 2.0f, M.diffuse_light(vec3(4, 4, 4)));
    w.add_quad(vec3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), M.diffuse_light(vec3(4, 4, 4)));

    s.cam.lookfrom = vec3(26, 3, 6);
    s.cam.lookat   = vec3(0, 2, 0);
    s.cam.vfov     = 20.0f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
}

inline void scene_cornell(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    const int red   = M.lambertian(vec3(.65f, .05f, .05f));
    const int blue  = M.lambertian(vec3(.15f, .15f, .75f));
    const int white = M.lambertian(vec3(.73f, .73f, .73f));
    const int light = M.diffuse_light(vec3(15, 15, 15));

    w.add_quad(vec3(0, 0, 0),       vec3(0, 555, 0),  vec3(0, 0, 555),  blue,  true);   // left
    w.add_quad(vec3(555, 0, 555),   vec3(0, 555, 0),  vec3(0, 0, -555), red,   true);   // right
    w.add_quad(vec3(0, 0, 0),       vec3(555, 0, 0),  vec3(0, 0, 555),  white, true);   // floor
    w.add_quad(vec3(0, 555, 555),   vec3(555, 0, 0),  vec3(0, 0, -555), white, true);   // ceiling
    w.add_quad(vec3(555, 0, 555),   vec3(-555, 0, 0), vec3(0, 555, 0),  white, true);   // back
    w.add_quad(vec3(213, 554, 227), vec3(130, 0, 0),  vec3(0, 0, 105),  light, true);   // light

    // The canonical Cornell boxes. The rotation and translation are baked into
    // each face's corner and edge vectors here, rather than being undone on
    // every ray by an instance wrapper.
    w.add_box(vec3(0, 0, 0), vec3(165, 165, 165), white,
              Transform::rotate_y_degrees(-18.0f).then_translate(vec3(130, 0, 65)));
    w.add_box(vec3(0, 0, 0), vec3(165, 330, 165), white,
              Transform::rotate_y_degrees(15.0f).then_translate(vec3(265, 0, 295)));

    // Hollow glass bubble: a sphere plus a slightly smaller one of negative
    // radius, which inverts its normals.
    const int glass = M.dielectric(1.5f);
    w.add_sphere(vec3(278, 335, 150),  60.0f, glass);
    w.add_sphere(vec3(278, 335, 150), -59.0f, glass);

    s.cam.lookfrom = vec3(278, 278, -800);
    s.cam.lookat   = vec3(278, 278, 0);
    s.cam.vfov     = 40.0f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
}

inline void scene_cornell_smoke(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    const int red   = M.lambertian(vec3(.65f, .05f, .05f));
    const int white = M.lambertian(vec3(.73f, .73f, .73f));
    const int green = M.lambertian(vec3(.12f, .45f, .15f));
    const int light = M.diffuse_light(vec3(7, 7, 7));

    w.add_quad(vec3(555, 0, 0),     vec3(0, 555, 0), vec3(0, 0, 555), green, true);
    w.add_quad(vec3(0, 0, 0),       vec3(0, 555, 0), vec3(0, 0, 555), red,   true);
    w.add_quad(vec3(0, 555, 0),     vec3(555, 0, 0), vec3(0, 0, 555), white, true);
    w.add_quad(vec3(0, 0, 0),       vec3(555, 0, 0), vec3(0, 0, 555), white, true);
    w.add_quad(vec3(0, 0, 555),     vec3(555, 0, 0), vec3(0, 555, 0), white, true);
    w.add_quad(vec3(113, 554, 127), vec3(330, 0, 0), vec3(0, 0, 305), light, true);

    w.add_medium_box(vec3(0, 0, 0), vec3(165, 330, 165), 0.01f, M.isotropic(vec3(0.5f, 0.5f, 0.5f)),
                     Transform::rotate_y_degrees(15.0f).then_translate(vec3(265, 0, 295)));
    w.add_medium_box(vec3(0, 0, 0), vec3(165, 165, 165), 0.01f, M.isotropic(vec3(1, 1, 1)),
                     Transform::rotate_y_degrees(-18.0f).then_translate(vec3(130, 0, 65)));

    s.cam.lookfrom = vec3(278, 278, -800);
    s.cam.lookat   = vec3(278, 278, 0);
    s.cam.vfov     = 40.0f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
}

// The 20x20 box floor shared by the two final scenes.
inline void add_final_ground(SceneBuilder& w, int ground_mat) {
    const int S = 20;
    for (int ix = 0; ix < S; ++ix) {
        for (int iz = 0; iz < S; ++iz) {
            const float width = 100.0f;
            const float x0 = -1000.0f + ix * width;
            const float z0 = -1000.0f + iz * width;
            // Stable pseudo-random height, so the scene is identical run to run.
            const float y1 = 1.0f + 100.0f * ((ix * 13 + iz * 37) % 100) / 100.0f;
            w.add_box(vec3(x0, 0, z0), vec3(x0 + width, y1, z0 + width), ground_mat);
        }
    }
}

// The 1000-sphere cluster, positioned by the same deterministic hash the
// device version used, so the layout is unchanged.
inline void add_final_cluster(SceneBuilder& w, int mat) {
    for (int j = 0; j < 1000; ++j) {
        vec3 p = random_in_unit_cube(j) * 165.0f;
        const float rad = 15.0f * (rt::pi / 180.0f);
        const float c = std::cos(rad), sn = std::sin(rad);
        p = vec3(c * p.x() + sn * p.z(), p.y(), -sn * p.x() + c * p.z()) + vec3(-100, 270, 395);
        w.add_sphere(p, 10.0f, mat);
    }
}

inline void scene_final(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    s.images.push_back("textures/earthmap.jpg");

    const int white  = M.lambertian(vec3(.73f, .73f, .73f));
    const int ground = M.lambertian(vec3(0.48f, 0.83f, 0.53f));
    const int light  = M.diffuse_light(vec3(7, 7, 7));

    add_final_ground(w, ground);
    w.add_quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light, true);

    const vec3 c1(400, 400, 200);
    w.add_sphere(c1, c1 + vec3(30, 0, 0), 50.0f, M.lambertian(vec3(0.7f, 0.3f, 0.1f)));

    w.add_sphere(vec3(260, 150, 45), 50.0f, M.dielectric(1.5f));
    w.add_sphere(vec3(0, 150, 145),  50.0f, M.metal(vec3(0.8f, 0.8f, 0.9f), 1.0f));

    // Glass shell that stays visible, with blue fog filling it.
    w.add_sphere(vec3(360, 150, 145), 70.0f, M.dielectric(1.5f));
    w.add_medium_sphere(vec3(360, 150, 145), 70.0f, 0.2f, M.isotropic(vec3(0.2f, 0.4f, 0.9f)));

    // Global thin white fog over the whole scene.
    w.add_medium_sphere(vec3(0, 0, 0), 5000.0f, 0.0001f, M.isotropic(vec3(1, 1, 1)));

    w.add_sphere(vec3(400, 200, 400), 100.0f, M.lambertian(M.image(0)));
    w.add_sphere(vec3(220, 280, 300),  80.0f, M.lambertian(M.noise(0.2f)));
    add_final_cluster(w, white);

    s.cam.lookfrom = vec3(478, 278, -600);
    s.cam.lookat   = vec3(278, 278, 0);
    s.cam.vfov     = 40.0f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
}

inline void scene_original(SceneSetup& s) {
    auto& w = s.world;
    auto& M = w.materials();

    s.images.push_back("textures/porcelain.jpg");   // image 0 (unused decal slot)
    s.images.push_back("textures/8ball.jpg");       // image 1

    const int white  = M.lambertian(vec3(.73f, .73f, .73f));
    const int ground = M.lambertian(vec3(0.88f, 0.50f, 0.76f));
    const int light  = M.diffuse_light(vec3(7, 7, 7));

    add_final_ground(w, ground);
    w.add_quad(vec3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light, true);

    const vec3 c1(400, 400, 200);
    w.add_sphere(c1, c1 + vec3(30, 0, 0), 50.0f,
                 M.lambertian(vec3(0.0488f, 0.0148f, 0.0171f)));

    w.add_sphere(vec3(260, 150, 45), 50.0f, M.dielectric(1.5f));
    w.add_sphere(vec3(0, 150, 145),  50.0f, M.metal(vec3(0.6387f, 0.3605f, 0.8826f), 1.0f));

    // 8-ball: textured core under a thin dielectric clear coat.
    w.add_sphere(vec3(360, 150, 145), 70.0f, M.lambertian(M.image(1)));
    w.add_sphere(vec3(360, 150, 145), 70.5f, M.dielectric(1.5f));

    w.add_medium_sphere(vec3(0, 0, 0), 5000.0f, 0.0001f, M.isotropic(vec3(1, 1, 1)));

    w.add_sphere(vec3(400, 200, 400), 100.0f, M.metal(vec3(0.23f, 0.24f, 0.85f), 0.02f));
    w.add_sphere(vec3(220, 280, 300),  80.0f, M.lambertian(M.noodle(0.2f)));
    add_final_cluster(w, white);

    s.cam.lookfrom = vec3(478, 278, -600);
    s.cam.lookat   = vec3(278, 278, 0);
    s.cam.vfov     = 40.0f;
    s.cam.focus_dist = (s.cam.lookfrom - s.cam.lookat).length();
    s.background = vec3(0.043f, 0.030f, 0.094f);
}

// ---------------------------------------------------------------------------

struct SceneInfo {
    SceneId     id;
    const char* name;
    int         width, height, spp;
};

inline const SceneInfo* scene_table() {
    static const SceneInfo t[SCENE_COUNT] = {
        { SCENE_BOUNCING,      "bouncing",      1200, 600, 10000 },
        { SCENE_CHECKERED,     "checkered",     1200, 600,   500 },
        { SCENE_EARTH,         "earth",         1200, 600,   500 },
        { SCENE_PERLIN,        "perlin",        1200, 600,   500 },
        { SCENE_QUADS,         "quads",         1200, 600,   500 },
        { SCENE_SIMPLE_LIGHT,  "simple_light",  1200, 600, 10000 },
        { SCENE_CORNELL,       "cornell",        600, 600, 10000 },
        { SCENE_CORNELL_SMOKE, "cornell_smoke",  600, 600,  1000 },
        { SCENE_FINAL,         "final",          800, 800, 10000 },
        { SCENE_ORIGINAL,      "original",       800, 800, 10000 },
    };
    return t;
}

inline void build_scene(SceneId id, unsigned int seed, SceneSetup& s) {
    switch (id) {
        case SCENE_BOUNCING:      scene_bouncing(s, seed);  break;
        case SCENE_CHECKERED:     scene_checkered(s);       break;
        case SCENE_EARTH:         scene_earth(s);           break;
        case SCENE_PERLIN:        scene_perlin(s);          break;
        case SCENE_QUADS:         scene_quads(s);           break;
        case SCENE_SIMPLE_LIGHT:  scene_simple_light(s);    break;
        case SCENE_CORNELL:       scene_cornell(s);         break;
        case SCENE_CORNELL_SMOKE: scene_cornell_smoke(s);   break;
        case SCENE_FINAL:         scene_final(s);           break;
        case SCENE_ORIGINAL:      scene_original(s);        break;
        default: break;
    }
    s.world.build_bvh();
}

} // namespace rt
