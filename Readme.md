## Data Helper
Using for process the robot tactile datasets
### usage
```bash
python setup.py develop
```

```bash
zsh run.sh process_conf/20231211_video2img.yaml
```

### newest version
```bash
# use 20240208_gen_depth_force.yaml to gen depth3D, depth2D, force_data
zsh run.sh process_conf/20240208_gen_depth_force.yaml
# use 20231212_tactile_learner.yaml to gen flow, uv_flow
zsh run.sh process_conf/20231212_tactile_learner.yaml
```

### data format
```bash
└─datasets
    └─apple
        ├─depth3D
        │  └─GelSightR_1707330955058305
        │    ├─depth3D_0.jpg
        │    ├─...
        │    └─depth3D_x.jpg
        │
        ├─final_meta
        │  └─GelSightL_1707330955058305_3_55
        │    ├─depth3D#20.mp4
        │    ├─flow#20.mp4
        │    ├─force2D#20.mp4
        │    ├─rgb#20.mp4
        │    └─uv_flow#20.mp4
        │
        ├─flow
        │  └─GelSightR_1707330955058305
        │    ├─flow_0.jpg
        │    ├─...
        │    └─flow_x.jpg
        │
        ├─force_data
        │  └─GelSightR_1707330955058305.json
        │
        ├─force_uv_data
        │  └─GelSightR_1707330955058305.json
        │
        ├─raw_meta
        │  └─GelSightR_1707330955058305.mp4
        │
        ├─rgb
        │  └─GelSightR_1707330955058305
        │    ├─rgb_0.jpg
        │    ├─...
        │    └─rgb_x.jpg
        │
        └─uv_flow
           └─GelSightR_1707330955058305
             ├─uv_flow_0.jpg
             ├─...
             └─uv_flow_x.jpg
```