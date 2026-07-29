/*
 * Headless driver for the reference AVBD implementation.
 *
 * This links against the UNMODIFIED solver sources from the authors' 3D demo
 * (github.com/savant117/avbd-demo3d), with only two mechanical changes applied
 * by build_reference.sh:
 *
 *   1. the OpenGL / Windows includes are stripped from solver.h, since we never
 *      render anything here;
 *   2. `float` is widened to `double` throughout, so the reference runs at the
 *      same precision as JavaScript numbers. Without this, every comparison
 *      would be swamped by float-vs-double rounding rather than showing whether
 *      the two implementations agree.
 *
 * No solver logic is touched. The program loads a scene file, steps the solver
 * a fixed number of times, and prints full body state so the JavaScript port
 * can be diffed against it.
 */

#include "solver.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace
{

struct SceneSpec
{
    std::vector<Rigid *> bodies;
};

double parseValue(const char *token)
{
    if (std::strcmp(token, "inf") == 0)
        return INFINITY;
    if (std::strcmp(token, "-inf") == 0)
        return -INFINITY;
    return std::strtod(token, nullptr);
}

bool loadScene(Solver *solver, const char *path, SceneSpec &spec)
{
    FILE *file = std::fopen(path, "r");
    if (!file)
    {
        std::fprintf(stderr, "cannot open scene file: %s\n", path);
        return false;
    }

    char line[4096];
    while (std::fgets(line, sizeof(line), file))
    {
        // Strip comments and skip blank lines
        char *hash = std::strchr(line, '#');
        if (hash)
            *hash = '\0';

        std::vector<char *> tok;
        for (char *t = std::strtok(line, " \t\r\n"); t; t = std::strtok(nullptr, " \t\r\n"))
            tok.push_back(t);

        if (tok.empty())
            continue;

        if (std::strcmp(tok[0], "body") == 0)
        {
            // sx sy sz density friction px py pz vx vy vz qx qy qz qw wx wy wz
            double v[18];
            for (int i = 0; i < 18; i++)
                v[i] = parseValue(tok[1 + i]);

            Rigid *body = new Rigid(solver,
                                    float3{v[0], v[1], v[2]},
                                    v[3], v[4],
                                    float3{v[5], v[6], v[7]},
                                    float3{v[8], v[9], v[10]});

            body->positionAng = quat{v[11], v[12], v[13], v[14]};
            body->initialAng = body->positionAng;
            body->velocityAng = float3{v[15], v[16], v[17]};

            spec.bodies.push_back(body);
        }
        else if (std::strcmp(tok[0], "joint") == 0)
        {
            // idxA idxB rax ray raz rbx rby rbz stiffLin stiffAng fracture
            int ia = (int)parseValue(tok[1]);
            int ib = (int)parseValue(tok[2]);
            double v[9];
            for (int i = 0; i < 9; i++)
                v[i] = parseValue(tok[3 + i]);

            new Joint(solver,
                      ia < 0 ? nullptr : spec.bodies[ia],
                      spec.bodies[ib],
                      float3{v[0], v[1], v[2]},
                      float3{v[3], v[4], v[5]},
                      v[6], v[7], v[8]);
        }
        else if (std::strcmp(tok[0], "spring") == 0)
        {
            // idxA idxB rax ray raz rbx rby rbz stiffness rest
            int ia = (int)parseValue(tok[1]);
            int ib = (int)parseValue(tok[2]);
            double v[8];
            for (int i = 0; i < 8; i++)
                v[i] = parseValue(tok[3 + i]);

            new Spring(solver,
                       spec.bodies[ia],
                       spec.bodies[ib],
                       float3{v[0], v[1], v[2]},
                       float3{v[3], v[4], v[5]},
                       v[6], v[7]);
        }
        else
        {
            std::fprintf(stderr, "unknown record: %s\n", tok[0]);
            std::fclose(file);
            return false;
        }
    }

    std::fclose(file);
    return true;
}

void dumpState(const SceneSpec &spec, int step)
{
    for (size_t i = 0; i < spec.bodies.size(); i++)
    {
        const Rigid *b = spec.bodies[i];
        std::printf(
            "%d %zu %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            step, i,
            b->positionLin.x, b->positionLin.y, b->positionLin.z,
            b->positionAng.x, b->positionAng.y, b->positionAng.z, b->positionAng.w,
            b->velocityLin.x, b->velocityLin.y, b->velocityLin.z,
            b->velocityAng.x, b->velocityAng.y, b->velocityAng.z);
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr,
                     "usage: %s <scene file> [steps] [iterations] [dt] [gravity] "
                     "[alpha] [betaLin] [betaAng] [gamma]\n",
                     argv[0]);
        return 1;
    }

    Solver solver;
    solver.defaultParams();

    const int steps = argc > 2 ? std::atoi(argv[2]) : 120;
    if (argc > 3) solver.iterations = std::atoi(argv[3]);
    if (argc > 4) solver.dt = parseValue(argv[4]);
    if (argc > 5) solver.gravity = parseValue(argv[5]);
    if (argc > 6) solver.alpha = parseValue(argv[6]);
    if (argc > 7) solver.betaLin = parseValue(argv[7]);
    if (argc > 8) solver.betaAng = parseValue(argv[8]);
    if (argc > 9) solver.gamma = parseValue(argv[9]);

    SceneSpec spec;
    if (!loadScene(&solver, argv[1], spec))
        return 1;

    for (int step = 1; step <= steps; step++)
    {
        solver.step();
        dumpState(spec, step);
    }

    return 0;
}
