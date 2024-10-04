# Tooth Segmentation Network (TSGNet)

conda create --name tsgnet python=3.9

conda activate tsgnet 

# install requirements 

* conda install git 

* python  install_packages.py

* git clone https://github.com/limhoyeon/ToothGroupNetwork.git 


# 1.Prediction of a mesh :: Input : folder contains upper and lower .obj mesh ==> Output : 2 Jsons files  each upper and Lower labled 



# 2.visualisation of the segmentation result upper/Lower  

python eval_visualize_results.py --mesh_path input_Mesh/013FHA7K/013FHA7K_upper.obj --pred_json_path result_json/013FHA7K_upper.json
