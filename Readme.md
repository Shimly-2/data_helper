## Data Helper
Using for process the robot tactile datasets
### usage
```bash
conda create -n data_helper python==3.8
python setup.py develop
pip --default-timeout=1000 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -r requirements.txt
```

### tactile version
```bash
zsh run.sh process_conf/20240111_gen_tactile_dataset.yaml
```

### input data format
```bash
└─datasets
    └─[task_name]
        ├─[case_name_1]
        │  └─raw_meta
        │     ├─[object_name_1]
        │     │  ├─[epoch_0]
        │     │  │  ├─front.mp4
        │     │  │  ├─fronttop.mp4
        │     │  │  ├─root.mp4
        │     │  │  ├─side.mp4
        │     │  │  ├─topdown.mp4
        │     │  │  ├─wrist.mp4
        │     │  │  ├─gelsightL.mp4
        │     │  │  ├─gelsightR.mp4
        │     │  │  └─result.json
        │     │  ├─...
        │     │  └─[epoch_x]
        │     ├─...
        │     └─[object_name_x]
        ├─...
        └─[case_name_x]
```

### output data format
```bash
└─datasets
    └─[task_name]
        ├─[case_name_1]
        │  └─train_meta
        │  │  ├─[object_name_1]
        │  │  │  ├─[cam_view_0]        
        │  │  │  │  ├─[epoch_0]
        │  │  │  │  │  └─rgb
        │  │  │  │  │  │  ├─rgb_0.png
        │  │  │  │  │  │  ├─...
        │  │  │  │  │  │  └─rgb_x.png
        │  │  │  │  ├─...
        │  │  │  │  └─[epoch_x]
        │  │  │  ├─...
        │  │  │  └─[cam_view_x]
        │  │  ├─...
        │  │  ├─[object_name_x]
        │  │  └─dataset_info.json
        │  └─vis_meta
        │     ├─epoch_0.mp4
        │     ├─...
        │     └─epoch_x.mp4
        ├─...
        └─[case_name_x]
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

### version for tactile2clip
```bash
# use multi_window_split.yaml to gen window and train set
zsh run.sh process_conf/multi_window_split.yaml
```

### input data format
```bash
└─datasets
    └─[task_name]
        ├─[case_name_1]
        │  └─raw_meta
        │     ├─[object_name_1]
        │     │  ├─[epoch_0]
        │     │  │  ├─front.mp4
        │     │  │  ├─fronttop.mp4
        │     │  │  ├─root.mp4
        │     │  │  ├─side.mp4
        │     │  │  ├─topdown.mp4
        │     │  │  ├─wrist.mp4
        │     │  │  ├─gelsightL.mp4
        │     │  │  ├─gelsightR.mp4
        │     │  │  └─result.json
        │     │  ├─...
        │     │  └─[epoch_x]
        │     ├─...
        │     └─[object_name_x]
        ├─...
        └─[case_name_x]
```

### output data format
```bash
└─datasets
    └─[task_name]
        ├─[case_name_1]
        │  ├─raw_meta
        │  │  ├─[object_name_1]
        │  │  │  ├─[epoch_0]
        │  │  │  │  ├─front.mp4
        │  │  │  │  ├─front#s224.mp4
        │  │  │  │  ├─fronttop.mp4
        │  │  │  │  ├─fronttop#s224.mp4
        │  │  │  │  ├─root.mp4
        │  │  │  │  ├─side.mp4
        │  │  │  │  ├─topdown.mp4
        │  │  │  │  ├─wrist.mp4
        │  │  │  │  ├─gelsightL.mp4
        │  │  │  │  ├─gelsightL#s224.mp4
        │  │  │  │  ├─gelsightR.mp4
        │  │  │  │  ├─gelsightR#s224.mp4
        │  │  │  │  └─result.json
        │  │  │  │  └─window_set.json
        │  │  │  ├─...
        │  │  │  └─[epoch_x]
        │  │  ├─...
        │  │  └─[object_name_x]
        │  ├─train_sec_{}_seed_{}.json
        │  ├─val_sec_{}_seed_{}.json
        │  └─test_sec_{}_seed_{}.json
        ├─...
        └─[case_name_x]
```