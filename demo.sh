python3 main_icra21.py \
  --train 0 --batch_size 1 --dataset_type demo --window -10 -5 0 5 10 \
  --dataset_pickle_file ./pickles/scannet_rob_test.pkl \
  --raft_model ./checkpoints/raft.ckpt \
  --drn_model ./checkpoints/drn.ckpt \
  --uncertainty_threshold -1 --refinement_iterations 7 \
  --save_visualization ./visualization
