Search.setIndex({docnames:["README","implemented_metrics","index","interfaces","interfaces.UI","modules","molSim","molSim.chemical_datastructures","molSim.ops","molSim.tasks","molSim.utils","setup","test","tests"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["README.rst","implemented_metrics.rst","index.rst","interfaces.rst","interfaces.UI.rst","modules.rst","molSim.rst","molSim.chemical_datastructures.rst","molSim.ops.rst","molSim.tasks.rst","molSim.utils.rst","setup.rst","test.rst","tests.rst"],objects:{"":[[3,0,0,"-","interfaces"],[6,0,0,"-","molSim"],[12,0,0,"-","test"],[13,0,0,"-","tests"]],"interfaces.config_reader":[[3,1,1,"","main"]],"molSim.chemical_datastructures":[[7,0,0,"-","molecule"],[7,0,0,"-","molecule_set"]],"molSim.chemical_datastructures.molecule":[[7,2,1,"","Molecule"]],"molSim.chemical_datastructures.molecule.Molecule":[[7,3,1,"","draw"],[7,3,1,"","get_descriptor_val"],[7,3,1,"","get_mol_property_val"],[7,3,1,"","get_name"],[7,3,1,"","get_similarity_to"],[7,3,1,"","is_same"],[7,3,1,"","match_fingerprint_from"],[7,3,1,"","set_descriptor"]],"molSim.chemical_datastructures.molecule_set":[[7,2,1,"","MoleculeSet"]],"molSim.chemical_datastructures.molecule_set.MoleculeSet":[[7,3,1,"","cluster"],[7,3,1,"","compare_against_molecule"],[7,3,1,"","get_cluster_labels"],[7,3,1,"","get_distance_matrix"],[7,3,1,"","get_mol_names"],[7,3,1,"","get_mol_properties"],[7,3,1,"","get_most_dissimilar_pairs"],[7,3,1,"","get_most_similar_pairs"],[7,3,1,"","get_pairwise_similarities"],[7,3,1,"","get_property_of_most_dissimilar"],[7,3,1,"","get_property_of_most_similar"],[7,3,1,"","get_similarity_matrix"],[7,3,1,"","get_transformed_descriptors"],[7,3,1,"","is_present"]],"molSim.exceptions":[[6,4,1,"","InvalidConfigurationError"],[6,4,1,"","LoadingError"],[6,4,1,"","MordredCalculatorError"],[6,4,1,"","NotInitializedError"]],"molSim.ops":[[8,0,0,"-","clustering"],[8,0,0,"-","descriptor"],[8,0,0,"-","similarity_measures"]],"molSim.ops.clustering":[[8,2,1,"","Cluster"]],"molSim.ops.clustering.Cluster":[[8,3,1,"","fit"],[8,3,1,"","get_labels"],[8,3,1,"","predict"]],"molSim.ops.descriptor":[[8,2,1,"","Descriptor"]],"molSim.ops.descriptor.Descriptor":[[8,3,1,"","check_init"],[8,3,1,"","fold_to_equal_length"],[8,3,1,"","get_all_supported_descriptors"],[8,3,1,"","get_folded_fprint"],[8,3,1,"","get_label"],[8,3,1,"","get_params"],[8,3,1,"","get_supported_fprints"],[8,3,1,"","is_fingerprint"],[8,3,1,"","make_fingerprint"],[8,3,1,"","set_manually"],[8,3,1,"","shorten_label"],[8,3,1,"","to_numpy"],[8,3,1,"","to_rdkit"]],"molSim.ops.similarity_measures":[[8,2,1,"","SimilarityMeasure"]],"molSim.ops.similarity_measures.SimilarityMeasure":[[8,3,1,"","get_compatible_metrics"],[8,3,1,"","get_supported_binary_metrics"],[8,3,1,"","get_supported_general_metrics"],[8,3,1,"","get_supported_metrics"],[8,3,1,"","get_uniq_metrics"],[8,3,1,"","is_distance_metric"]],"molSim.tasks":[[9,0,0,"-","cluster_data"],[9,0,0,"-","compare_target_molecule"],[9,0,0,"-","identify_outliers"],[9,0,0,"-","measure_search"],[9,0,0,"-","see_property_variation_with_similarity"],[9,0,0,"-","task"],[9,0,0,"-","task_manager"],[9,0,0,"-","visualize_dataset"]],"molSim.tasks.cluster_data":[[9,2,1,"","ClusterData"]],"molSim.tasks.compare_target_molecule":[[9,2,1,"","CompareTargetMolecule"]],"molSim.tasks.compare_target_molecule.CompareTargetMolecule":[[9,3,1,"","get_hits_dissimilar_to"],[9,3,1,"","get_hits_similar_to"]],"molSim.tasks.identify_outliers":[[9,2,1,"","IdentifyOutliers"]],"molSim.tasks.measure_search":[[9,2,1,"","MeasureSearch"]],"molSim.tasks.measure_search.MeasureSearch":[[9,3,1,"","get_best_measure"]],"molSim.tasks.see_property_variation_with_similarity":[[9,2,1,"","SeePropertyVariationWithSimilarity"]],"molSim.tasks.see_property_variation_with_similarity.SeePropertyVariationWithSimilarity":[[9,3,1,"","get_property_correlations_in_most_dissimilar"],[9,3,1,"","get_property_correlations_in_most_similar"]],"molSim.tasks.task":[[9,2,1,"","Task"]],"molSim.tasks.task_manager":[[9,2,1,"","TaskManager"]],"molSim.tasks.visualize_dataset":[[9,2,1,"","VisualizeDataset"]],"molSim.utils":[[10,0,0,"-","ccbmlib_fingerprints"],[10,0,0,"-","plotting_scripts"]],"molSim.utils.ccbmlib_fingerprints":[[10,1,1,"","atom_pairs"],[10,1,1,"","avalon"],[10,1,1,"","generate_fingerprints"],[10,1,1,"","hash_parameter_set"],[10,1,1,"","hashed_atom_pairs"],[10,1,1,"","hashed_morgan"],[10,1,1,"","hashed_torsions"],[10,1,1,"","maccs_keys"],[10,1,1,"","morgan"],[10,1,1,"","rdkit_fingerprint"],[10,1,1,"","to_key_val_string"],[10,1,1,"","torsions"]],"molSim.utils.plotting_scripts":[[10,1,1,"","plot_barchart"],[10,1,1,"","plot_density"],[10,1,1,"","plot_heatmap"],[10,1,1,"","plot_multiple_barchart"],[10,1,1,"","plot_parity"],[10,1,1,"","plot_scatter"]],"tests.test_CompareTargetMolecule":[[13,2,1,"","TestCompareTargetMolecule"]],"tests.test_CompareTargetMolecule.TestCompareTargetMolecule":[[13,3,1,"","smiles_seq_to_xl_or_csv"],[13,3,1,"","tearDownClass"],[13,3,1,"","test_get_molecule_least_similar_to"],[13,3,1,"","test_get_molecule_most_similar_to"]],"tests.test_Descriptor":[[13,2,1,"","TestDescriptor"]],"tests.test_Descriptor.TestDescriptor":[[13,3,1,"","test_bad_descriptors_padelpy_descriptors"],[13,3,1,"","test_descriptor_arbitrary_list_init"],[13,3,1,"","test_descriptor_arbitrary_numpy_init"],[13,3,1,"","test_descriptor_empty_init"],[13,3,1,"","test_descriptor_make_fingerprint"],[13,3,1,"","test_fingerprint_folding"],[13,3,1,"","test_mordred_descriptors"],[13,3,1,"","test_nonexistent_mordred_descriptors"],[13,3,1,"","test_padelpy_descriptors"],[13,3,1,"","test_topological_fprint_min_path_lesser_than_atoms"]],"tests.test_LoadingErrorException":[[13,2,1,"","TestLoadingERrorException"]],"tests.test_LoadingErrorException.TestLoadingERrorException":[[13,3,1,"","test_invalid_pdb"],[13,3,1,"","test_invalid_smiles"],[13,3,1,"","test_missing_pdb"],[13,3,1,"","test_missing_smiles"]],"tests.test_MeasureSearch":[[13,2,1,"","TestMeasureSearch"]],"tests.test_MeasureSearch.TestMeasureSearch":[[13,3,1,"","smiles_seq_to_textfile"],[13,3,1,"","test_error_optim_algo"],[13,3,1,"","test_fixed_SimilarityMeasure"],[13,3,1,"","test_fixed_fprint"],[13,3,1,"","test_logfile_generation"],[13,3,1,"","test_max_optim_algo"],[13,3,1,"","test_min_optim_algo"],[13,3,1,"","test_msearch_completion"],[13,3,1,"","test_msearch_init"],[13,3,1,"","test_msearch_init_error"],[13,3,1,"","test_only_metric_search"],[13,5,1,"","test_smiles"],[13,3,1,"","test_verbose_output"]],"tests.test_Molecule":[[13,2,1,"","TestMolecule"]],"tests.test_Molecule.TestMolecule":[[13,3,1,"","test_get_name"],[13,3,1,"","test_get_property_value"],[13,3,1,"","test_is_same"],[13,3,1,"","test_match_fprint_error"],[13,3,1,"","test_mol_mol_similarity_w_morgan_tanimoto"],[13,3,1,"","test_mol_smiles_loadingerror"],[13,3,1,"","test_mol_src_pdb_loadingerror"],[13,3,1,"","test_mol_src_txt_loadingerror"],[13,3,1,"","test_molecule_created_w_attributes"],[13,3,1,"","test_molecule_created_with_no_attributes"],[13,3,1,"","test_molecule_draw"],[13,3,1,"","test_molecule_graph_similar_to_itself_morgan_dice"],[13,3,1,"","test_molecule_graph_similar_to_itself_morgan_l0"],[13,3,1,"","test_molecule_graph_similar_to_itself_morgan_tanimoto"],[13,3,1,"","test_set_molecule_from_file"],[13,3,1,"","test_set_molecule_from_smiles"]],"tests.test_MoleculeSet":[[13,2,1,"","TestMoleculeSet"]],"tests.test_MoleculeSet.TestMoleculeSet":[[13,3,1,"","smarts_seq_to_smiles_file"],[13,3,1,"","smiles_seq_to_pdb_dir"],[13,3,1,"","smiles_seq_to_smi_file"],[13,3,1,"","smiles_seq_to_smiles_file"],[13,3,1,"","smiles_seq_to_textfile"],[13,3,1,"","smiles_seq_to_xl_or_csv"],[13,3,1,"","test_clustering_fingerprints"],[13,3,1,"","test_get_most_dissimilar_pairs"],[13,3,1,"","test_get_most_similar_pairs"],[13,3,1,"","test_invalid_transform_error"],[13,3,1,"","test_mds_transform"],[13,3,1,"","test_molecule_set_getters"],[13,3,1,"","test_molecule_set_sim_getters"],[13,3,1,"","test_pca_transform"],[13,3,1,"","test_set_molecule_database_fingerprint_from_csv"],[13,3,1,"","test_set_molecule_database_from_csv"],[13,3,1,"","test_set_molecule_database_from_excel"],[13,3,1,"","test_set_molecule_database_from_pdb_dir"],[13,3,1,"","test_set_molecule_database_from_smarts_file"],[13,3,1,"","test_set_molecule_database_from_smi_file"],[13,3,1,"","test_set_molecule_database_from_smiles_file"],[13,3,1,"","test_set_molecule_database_from_textfile"],[13,3,1,"","test_set_molecule_database_w_descriptor_property_from_csv"],[13,3,1,"","test_set_molecule_database_w_descriptor_property_from_excel"],[13,3,1,"","test_set_molecule_database_w_fingerprint_similarity_from_csv"],[13,3,1,"","test_set_molecule_database_w_property_from_csv"],[13,3,1,"","test_set_molecule_database_w_property_from_excel"],[13,3,1,"","test_set_molecule_database_w_property_from_textfile"],[13,3,1,"","test_set_molecule_database_w_similarity_from_csv"],[13,5,1,"","test_smarts"],[13,5,1,"","test_smiles"],[13,3,1,"","test_subsample_molecule_database_from_csv"],[13,3,1,"","test_subsample_molecule_database_from_excel"],[13,3,1,"","test_subsample_molecule_database_from_pdb_dir"],[13,3,1,"","test_subsample_molecule_database_from_textfile"],[13,3,1,"","test_tsne_transform"]],"tests.test_SimilarityMeasure":[[13,2,1,"","TestSimilarityMeasure"]],"tests.test_SimilarityMeasure.TestSimilarityMeasure":[[13,3,1,"","smiles_seq_to_xl_or_csv"],[13,3,1,"","test_all_supported_measures"],[13,3,1,"","test_get_abcd"],[13,3,1,"","test_similarity_measure_limits"]],"tests.test_SimilarityMeasureValueErrors":[[13,2,1,"","TestSimilarityMeasureValueError"]],"tests.test_SimilarityMeasureValueErrors.TestSimilarityMeasureValueError":[[13,3,1,"","test_binary_only_metrics"],[13,3,1,"","test_compatible_metrics"],[13,3,1,"","test_empty_fprints"],[13,3,1,"","test_invalid_metric"],[13,3,1,"","test_vectornorm_length_errors"]],"tests.test_TaskManager":[[13,2,1,"","TestTaskManager"]],"tests.test_TaskManager.TestTaskManager":[[13,3,1,"","setUpClass"],[13,3,1,"","test_no_tasks_task_manager"],[13,3,1,"","test_task_manager"]],"tests.test_multithreading":[[13,2,1,"","TestMultithreading"]],"tests.test_multithreading.TestMultithreading":[[13,3,1,"","setUpClass"],[13,3,1,"","tearDownClass"],[13,3,1,"","test_multithreading_consistency_10_threads"],[13,3,1,"","test_multithreading_consistency_2_threads"],[13,3,1,"","test_multithreading_consistency_3_threads"],[13,3,1,"","test_multithreading_consistency_4_threads"],[13,3,1,"","test_multithreading_consistency_5_threads"],[13,3,1,"","test_multithreading_consistency_6_threads"],[13,3,1,"","test_multithreading_consistency_7_threads"],[13,3,1,"","test_speedup_efficiency_cosine"],[13,3,1,"","test_speedup_efficiency_tanimoto"]],interfaces:[[4,0,0,"-","UI"],[3,0,0,"-","config_reader"]],molSim:[[7,0,0,"-","chemical_datastructures"],[6,0,0,"-","exceptions"],[8,0,0,"-","ops"],[9,0,0,"-","tasks"],[10,0,0,"-","utils"]],tests:[[13,0,0,"-","test_CompareTargetMolecule"],[13,0,0,"-","test_Descriptor"],[13,0,0,"-","test_LoadingErrorException"],[13,0,0,"-","test_MeasureSearch"],[13,0,0,"-","test_Molecule"],[13,0,0,"-","test_MoleculeSet"],[13,0,0,"-","test_SimilarityMeasure"],[13,0,0,"-","test_SimilarityMeasureValueErrors"],[13,0,0,"-","test_TaskManager"],[13,0,0,"-","test_multithreading"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","exception","Python exception"],"5":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:exception","5":"py:attribute"},terms:{"0":[7,9,10,13],"01":9,"1":[0,1,7,10,13],"10":[0,1,13],"1002":0,"1021":0,"1038":0,"11":[1,13],"12":1,"13":1,"14":1,"140":0,"15":1,"16":1,"1669":0,"17":1,"18":1,"19":1,"2":[0,1,10,13],"20":[1,10],"2005":0,"2009":0,"2011":0,"2013":0,"2018":0,"2020":0,"21":1,"22":1,"23":1,"24":[1,10],"25":1,"26":1,"27":1,"28":[0,1],"29":1,"2nd":0,"3":[0,1,13],"30":1,"31":1,"32":1,"33":1,"34":1,"35":1,"36":1,"37":1,"38":1,"39":1,"4":[0,1,13],"40":1,"41":1,"42":[1,7],"43":1,"44":[0,1],"45":1,"46":1,"47":1,"5":[0,1,13],"53":0,"597":0,"6":[0,1,13],"601":0,"7":[0,1,13],"8":[0,1,7,13],"8781":0,"8787":0,"8b04532":0,"9":[1,13],"abstract":[7,9],"case":[0,7,10,13],"class":[6,7,8,9,13],"default":[7,8,9,10,13],"do":[0,13],"float":[7,9],"function":[10,13],"int":[7,8,9,10],"new":0,"return":[7,8,9,10,13],"static":[7,8],"throw":13,"true":[7,8,9],"try":[0,7,13],A:[0,7],AND:0,AS:0,BE:0,BUT:0,Be:0,FOR:0,For:0,IN:0,IS:0,If:[0,3,7,9,10],In:0,NO:0,NOT:0,OF:0,OR:0,THE:0,TO:0,The:[0,7,8,9],There:0,To:[0,10],WITH:0,__init__:0,_build:0,_can_:8,_fingerprint:8,abc:9,abil:13,abl:13,about:0,abov:0,absenc:7,absolut:9,accord:0,account:0,across:0,action:0,activ:0,addit:[0,7],addition:0,against:0,aggress:9,aka:8,al:9,algorithm:[0,7,9,13],alias:1,all:[0,7,8,9,13],allow:13,along:[0,10],alreadi:[0,13],also:0,alt:0,altern:0,although:9,am:0,an:[0,3,6,7,9,13],analysi:[7,13],ani:0,annot:10,anoth:[0,7],apidoc:0,appar:0,apparatu:0,appear:0,applic:0,approach:13,appropri:[0,9],ar:[0,7,8,9,10,13],arbitrari:[0,7,8,13],arbitrary_descriptor_v:[7,8],arg:[7,8,9,10,13],argument:[7,8,10],aris:0,arrai:[7,8,10,13],assess:0,associ:[0,7],atom_pair:10,attribut:[7,13],attributeerror:6,austin:1,author:0,autom:0,avail:0,avalon:10,averag:0,avoid:0,axi:10,b:[7,13],back:8,badge_logo:0,bar:10,baroni:1,base:[0,6,7,8,9,13],basic:13,becaus:7,becom:0,befor:13,behavior:13,being:[7,8],below:0,best:9,better:9,between:[0,7,8,9],bhattacharje:0,binari:[0,8],binder:0,bit:[8,13],blanquet:1,boil:7,bool:[7,8,9,10],borg:0,both:9,branch:0,braun:1,breviti:9,broken:7,browser:0,build:0,built:0,bump:0,burn:0,buser:1,c1:13,c:[0,13],calcul:[6,7,9],call:[0,3,6,8,9],campaign:0,can:[0,9,10,13],cannot:[6,13],carri:9,categori:10,cc1:13,cc:13,ccbmlib:0,ccbmlib_fingerprint:[5,6],ccc:13,cccc:13,ccccccc:13,cdatastruct:8,ch2:13,ch3:13,ch:13,charg:0,chart:10,check:[0,7,8,13],check_init:8,chem:0,chemic:0,chemical_datastructur:[5,6,9],chemist:0,chen:0,choi:1,choic:[7,9],choos:0,chosen:[0,7,9,13],cite:2,city_block_similar:1,claim:0,classmethod:13,closest:7,cluster:[0,5,6,7,9,13],cluster_data:[5,6],cluster_grouped_mol_nam:7,clusterdata:9,clustering_method:[7,8],cn:13,co:13,code:0,coeffici:7,cohen:1,cole:1,cole_1:1,cole_2:1,collect:7,collin:0,color:10,colwel:1,combin:13,combinatori:9,command:[0,3],common:10,commonli:[0,7],compar:[0,7,13],compare_against_molecul:7,compare_target_molecul:[5,6],comparetargetmolecul:[9,13],comparison:13,complet:[0,13],complex:[8,13],compon:[0,13],comprehens:0,compris:7,comptabil:8,condit:0,config:[0,9,13],config_read:5,configur:[3,6,9,13],confirgur:13,connect:0,consid:8,consist:13,consonni:1,consum:0,contain:[8,13],content:5,context:0,contract:0,contrera:0,contributor:2,control:7,convers:0,convert:[8,9,13],copi:0,copyright:0,core:0,correl:[0,9],correspond:9,cosin:1,coupl:0,cp:0,creat:[0,9,13],creation:13,csv:13,current:8,custom:13,d:0,damag:0,data:[0,7,9],databas:[0,9,13],dataset:[0,7,9],datastruct:8,davi:0,daylight:0,deal:0,decreas:9,defin:[0,9],delet:13,demo:0,denni:1,denot:[8,13],densiti:[0,10],desciptornam:0,descriptor:[0,5,6,7,13],descriptornam:0,design:0,desir:0,detect:0,determin:7,develop:2,diagon:0,dice:[1,13],dice_2:1,dice_3:1,dict:[7,8,9,10],dictionari:[7,8],differ:[0,10],dimens:0,dimension:[0,13],directli:[0,13],directori:[0,13],discov:0,discuss:0,dispers:1,displai:7,dissimilar:[0,7,9],dist:0,distanc:[0,1,7,8,9],distribut:0,divers:0,doc:0,doe:13,doi:0,don:[0,7],dotson:1,draw:[7,13],driver:1,drug:0,due:[0,9],duplic:7,dure:7,e:[7,9],each:[0,7,8,10],ecfp:0,ed:0,effici:0,efficieni:13,effort:0,either:7,element:0,empti:[3,13],enhanc:0,enough:0,ensur:[0,13],entir:0,equal:[8,10],equival:[0,7],erron:13,error:13,essenti:0,etc:[7,10],euclidean:0,euclidean_similar:1,evalu:[0,9,13],event:0,exampl:0,excel:13,except:[5,9,13],exclud:8,execut:[0,13],exist:[0,13],experiment:[0,8],explicitbitvect:8,exploratori:0,explos:9,express:0,f:0,fail:[6,7],faith:1,fals:[9,10],featur:[0,8,9,13],feature_arr:13,field:[3,9],file:[0,3,7,13],filenam:0,filetyp:13,find:[0,8,13],fingerprint1:8,fingerprint2:8,fingerprint:[7,8,9,13],fingerprint_param:[7,8],fingerprint_typ:[7,8,9,13],first:[7,10],fit:[0,8],fix:9,fixtur:13,fold:[8,13],fold_to_equal_length:8,fold_to_length:8,follow:0,fontsiz:10,forb:1,forest:0,form:[0,9],format:8,fossum:1,fp:[8,10],fpath:7,fraction:[7,9],free:0,friedman:0,from:[0,3,7,8,9,13],ftype:13,full:0,functionailti:13,furnish:0,further:0,furthest:[0,9],furthest_neighbor_correl:9,g:7,gener:[0,7],generate_fingerprint:10,generate_similarity_matrix:7,get:[7,8,9,13],get_all_supported_descriptor:8,get_best_measur:9,get_cluster_label:7,get_compatible_metr:[8,13],get_descriptor_v:7,get_distance_matrix:7,get_folded_fprint:8,get_hits_dissimilar_to:9,get_hits_similar_to:9,get_label:8,get_mol_nam:7,get_mol_properti:7,get_mol_property_v:7,get_molecule_least_similar_to:13,get_most_dissimilar_pair:[7,13],get_most_similar_pair:[7,13],get_nam:7,get_pairwise_similar:7,get_param:8,get_property_correlations_in_most_dissimilar:9,get_property_correlations_in_most_similar:9,get_property_of_most_dissimilar:7,get_property_of_most_similar:7,get_similarity_matrix:7,get_similarity_to:7,get_supported_binary_metr:8,get_supported_descriptor:8,get_supported_fprint:8,get_supported_general_metr:8,get_supported_metr:8,get_transformed_descriptor:7,get_uniq_metr:8,gh:0,github:0,give:0,gleason:1,gloriu:0,goldberg:1,good:0,goodman:1,goodman_krusk:1,grant:0,graph:[7,8,13],graphic:0,grid:10,groenen:0,group:0,ha:7,halid:0,harri:1,hash_parameter_set:10,hashed_atom_pair:10,hashed_morgan:10,hashed_tors:10,hasti:0,have:[7,9],hawkin:1,he:9,heatmap:[0,10],height:10,help:0,helper:13,herebi:0,heron:1,heteratom:0,hierarch:0,high:0,highest:9,himaghna:0,holder:0,holidai:1,holiday_denni:1,holiday_fossum:1,hook:13,hour:0,how:7,html:0,http:0,i:[0,7,9],id:[7,9],idea:0,ideal:13,ident:13,identif:13,identifi:[0,7,9],identify_outli:[5,6],identifyoutli:9,imag:[0,7],implement:[7,8,9,13],impli:0,in_dtyp:8,includ:0,index:[2,7],indic:[0,7,9,10],infer:0,inform:[0,7],initi:[6,13],input:[0,1,3,8,10,13],input_matrix:10,instal:2,instant:13,instanti:13,instead:0,interest:[0,9],interfac:[0,2,5],invalid:[6,13],invalidconfigurationerror:[6,8,10],invok:8,io:7,ioerror:3,ipynb:0,is_distance_metr:8,is_fingerprint:8,is_pres:7,is_sam:7,is_verbos:7,isol:0,isolationforest:9,issu:0,iter:0,its:[0,9],itself:13,j:0,jac:0,jaccard:1,jackson:0,just:10,k:0,keep:7,kei:7,keyword:[7,8,10],kind:0,knowl:0,kroeber:1,kruskal:1,kulczynski:1,kwarg:[7,8,9,10],l0:[0,13],l0_similar:1,l1:0,l1_similar:1,l2:0,l2_similar:1,lab:0,label:[7,8,9,10,13],label_:8,labori:0,labpath:0,lahei:1,lead:0,learn:0,least:[7,13],legend:10,legend_label:10,length:[8,13],less:10,level:7,liabil:0,liabl:0,librari:0,licens:2,limit:0,line:[0,3],linkag:0,linkedin:0,list:[0,7,8,9,10,13],load:[6,13],loadingerror:[6,13],log:13,longer:8,look:9,m2r:0,m:0,maccs_kei:10,machin:0,made:0,mai:7,main:3,make:[0,8],make_fingerprint:8,manhattan_similar:1,mani:0,manipul:7,manual:[0,8],master:0,match:13,match_fingerprint_from:7,matric:13,matrix:[7,8,10,13],max:9,max_min:9,maxim:[0,9],maximum:13,maxwel:1,md:[0,13],measur:[0,7,8,9,13],measure_search:[5,6],measuresearch:[9,13],medoid:0,merchant:0,merg:0,messag:6,method:[7,8,13],method_:7,methodnam:13,metric:[0,2,7,8,9,13],michael:1,michen:1,might:0,min:9,mine:0,minim:[0,9],minimum:13,miss:13,mit:0,model:[0,8],modern:0,modifi:[0,7,8,10],modul:[2,5],moieti:0,mol:[7,10,13],mol_descriptor_v:7,mol_graph:7,mol_property_v:7,mol_smil:[7,13],mol_src:[7,13],mol_suppl:10,mol_text:7,molecul:[0,5,6,8,9,13],molecular:[0,7,8,13],molecule_databas:7,molecule_database_src:7,molecule_database_src_typ:7,molecule_graph:8,molecule_set:[5,6,9],molecule_set_config:9,moleculeset:[7,9,13],moleculset:13,molsim:[3,13],molsim_ui_main:[3,5],mordr:[0,6,13],mordredcalculatorerror:6,more:[0,9,10,13],morgan:[0,10,13],most:[0,7,9,13],mountford:1,much:7,multi:0,multidimension:0,multipl:[7,9,10],multiplear:10,multiprocess:[0,13],multithread:13,murrai:0,murtagh:0,mv:0,mybind:0,n:[10,13],n_bar:10,n_bars_per_xtick:10,n_cluster:[7,8],n_densiti:10,n_mol:7,n_points_per_dens:10,n_thread:7,n_xtick:10,name:[1,7,10,13],name_seq:13,namedtupl:9,natur:0,nchem:0,ndarrai:[7,8,9,10,13],nearest:[0,9],nearest_neighbor_correl:9,need:[0,9],neighbor:[0,9],newli:0,nh:13,non:[7,13],none:[6,7,8,9,10,13],noninfring:0,norm:[0,13],note:[2,7,8],notic:0,notinitializederror:[6,7,13],novel:0,np:[7,8,9,10,13],num_hit:9,number:[0,7,8,10],numpi:[7,8,10,13],numpy_:8,o:[0,13],object:[6,7,8,9,13],obtain:0,ochiai:1,often:0,oh:13,one:[0,7,8,10],ones:[0,10],onli:[8,9,13],onlin:0,only_metr:9,op:[5,6,7],open:0,oper:0,optim:[0,9,13],optim_algo:9,option:[9,10,13],order:9,org:0,oserror:6,other:[0,7,8,13],otherwis:[0,13],our:0,out:[0,9],outlier:[0,9],outlier_idx:10,output:[0,6,13],over:[0,9],overal:9,overview:0,p:0,packag:[0,2,5],padelpi:13,pair:[0,7,13],pairwis:[0,7],par:10,paramet:[6,7,8,9],parent:9,pariti:10,particular:0,partner:0,pass:[7,10,13],passthrough:13,path:[7,13],pattern:0,pca:7,pdb:13,pearson:1,pearson_heron:1,peirc:1,peirce_1:1,peirce_2:1,per:10,perform:9,permiss:0,permit:0,person:0,pharmocolog:0,pillin:1,pip:0,plot:[0,7,8,9,10],plot_barchart:10,plot_dens:10,plot_heatmap:10,plot_multiple_barchart:10,plot_par:10,plot_scatt:10,plot_titl:10,plot_title_fonts:10,plotting_script:[5,6],point:7,portion:0,potenti:0,practic:0,predict:[0,8],present:7,princip:13,probabl:0,process:13,produc:13,project:0,proper:6,properti:[0,6,7,9,13],property_seq:13,propos:0,provid:0,publish:0,pull:0,purpos:[2,8],push:0,py:[0,6],pypi:0,pyplot:10,python:[0,13],qoi:0,qualiti:0,quantiti:9,queri:[7,9],query_molecul:[7,9],r:0,rais:[3,7,8,10,13],rand:1,randomli:[7,13],rao:1,rapid:0,rare:8,rdkit:[0,7,8],rdkit_:8,rdkit_fingerprint:10,reaction:0,read:[3,13],readm:2,realiti:0,recommend:9,reduct:13,redund:[0,8],refer:[0,7],reference_mol:7,region:0,relat:0,relev:7,remov:8,repertoir:0,replic:0,repres:[8,10],represent:[7,8],request:0,requir:0,research:0,respect:7,respons:[7,9,13],restrict:0,result:0,retriev:13,right:0,robust:0,roger:1,rogot:1,rst:0,run:[2,13],runtest:13,runtimeerror:6,russel:1,s:[0,7,8,10,13],same:[0,7,13],sampl:[7,10],sample_ratio:9,sampling_random_st:7,sampling_ratio:7,satisfi:9,scalar:13,scale:0,scatter:10,scheme:0,scope:0,score:[7,9],score_:9,screen:0,search:[0,7,13],second:[7,10],see:[0,7],see_property_variation_with_similar:[5,6],seepropertyvariationwithsimilar:9,select:[0,7,9],self:[7,9],sell:0,separ:[0,9],sequenc:[9,13],seri:[0,10],serial:13,set:[0,7,8,9,10,13],set_descriptor:7,set_manu:8,set_molecul:9,set_similar:7,setup:[2,5],setupclass:13,sever:0,shade:10,shall:0,shape:10,shortcom:0,shorten:8,shorten_label:8,should:[0,13],show_plot:10,show_top:[9,13],shown:0,similar:[2,7,8,9,10,13],similarity_matrix:7,similarity_measur:[5,6,7,9,13],similarity_scor:7,similaritymeasur:[7,8,13],simple_match:1,simpson:1,sinc:[7,9],singl:[0,13],site:0,size:[8,10],smaller:8,smart:13,smarts_seq_to_smiles_fil:13,smi:13,smile:13,smiles_seq_to_pdb_dir:13,smiles_seq_to_smi_fil:13,smiles_seq_to_smiles_fil:13,smiles_seq_to_textfil:13,smiles_seq_to_xl_or_csv:13,snake_similar:1,sneath:1,sneath_1:1,sneath_2:1,sneath_3:1,sneath_4:1,so:[0,13],soc:0,softwar:0,sokal:1,solvent:0,some:[7,10],sorenson:1,sorgenfrei:1,sort:9,sorted_similar:9,sourc:7,source_molecul:7,speci:0,specifi:[0,7,8,13],specifii:0,speedup:[0,13],spend:0,sphinx:0,springer:0,start:0,state:8,statist:0,step:0,stop:13,store:[7,13],str:[7,8,9,10,13],strategi:[8,9],string:[7,8,13],structur:0,studi:7,subclass:9,subject:0,sublicens:0,submodul:5,subpackag:5,subsampl:[9,13],subsample_subset_s:9,substanti:0,substitut:0,substrat:0,sulfon:0,sulfonamid:0,suppli:[7,8,9,10],support:[2,6],sure:0,susbtrat:0,svg:0,symmetr:1,symmetric_sokal_sneath:1,synthes:0,synthesi:0,t:[0,7],tag:[0,8],take:0,tanimoto:[1,13],target:[0,7],target_mol:7,target_molecul:7,task:[0,3,5,6,13],task_manag:[5,6],taskmanag:[9,13],taxicab_similar:1,teardownclass:13,temporari:13,term:0,termin:0,test:[0,2,5],test_all_supported_measur:13,test_bad_descriptors_padelpy_descriptor:13,test_binary_only_metr:13,test_clustering_fingerprint:13,test_comparetargetmolecul:5,test_compatible_metr:13,test_descriptor:5,test_descriptor_arbitrary_list_init:13,test_descriptor_arbitrary_numpy_init:13,test_descriptor_empty_init:13,test_descriptor_make_fingerprint:13,test_empty_fprint:13,test_error_optim_algo:13,test_fingerprint_fold:13,test_fixed_fprint:13,test_fixed_similaritymeasur:13,test_get_abcd:13,test_get_molecule_least_similar_to:13,test_get_molecule_most_similar_to:13,test_get_most_dissimilar_pair:13,test_get_most_similar_pair:13,test_get_nam:13,test_get_property_valu:13,test_invalid_metr:13,test_invalid_pdb:13,test_invalid_smil:13,test_invalid_transform_error:13,test_is_sam:13,test_loadingerrorexcept:5,test_logfile_gener:13,test_match_fprint_error:13,test_max_optim_algo:13,test_mds_transform:13,test_measuresearch:5,test_min_optim_algo:13,test_missing_pdb:13,test_missing_smil:13,test_mol_mol_similarity_w_morgan_tanimoto:13,test_mol_smiles_loadingerror:13,test_mol_src_pdb_loadingerror:13,test_mol_src_txt_loadingerror:13,test_molecul:5,test_molecule_created_w_attribut:13,test_molecule_created_with_no_attribut:13,test_molecule_draw:13,test_molecule_graph_similar_to_itself_morgan_dic:13,test_molecule_graph_similar_to_itself_morgan_l0:13,test_molecule_graph_similar_to_itself_morgan_tanimoto:13,test_molecule_set_gett:13,test_molecule_set_sim_gett:13,test_moleculeset:5,test_mordred_descriptor:13,test_msearch_complet:13,test_msearch_init:13,test_msearch_init_error:13,test_multithread:5,test_multithreading_consistency_10_thread:13,test_multithreading_consistency_2_thread:13,test_multithreading_consistency_3_thread:13,test_multithreading_consistency_4_thread:13,test_multithreading_consistency_5_thread:13,test_multithreading_consistency_6_thread:13,test_multithreading_consistency_7_thread:13,test_no_tasks_task_manag:13,test_nonexistent_mordred_descriptor:13,test_only_metric_search:13,test_padelpy_descriptor:13,test_pca_transform:13,test_set_molecule_database_fingerprint_from_csv:13,test_set_molecule_database_from_csv:13,test_set_molecule_database_from_excel:13,test_set_molecule_database_from_pdb_dir:13,test_set_molecule_database_from_smarts_fil:13,test_set_molecule_database_from_smi_fil:13,test_set_molecule_database_from_smiles_fil:13,test_set_molecule_database_from_textfil:13,test_set_molecule_database_w_descriptor_property_from_csv:13,test_set_molecule_database_w_descriptor_property_from_excel:13,test_set_molecule_database_w_fingerprint_similarity_from_csv:13,test_set_molecule_database_w_property_from_csv:13,test_set_molecule_database_w_property_from_excel:13,test_set_molecule_database_w_property_from_textfil:13,test_set_molecule_database_w_similarity_from_csv:13,test_set_molecule_from_fil:13,test_set_molecule_from_smil:13,test_similarity_measure_limit:13,test_similaritymeasur:5,test_similaritymeasurevalueerror:5,test_smart:13,test_smil:13,test_speedup_efficiency_cosin:13,test_speedup_efficiency_tanimoto:13,test_subsample_molecule_database_from_csv:13,test_subsample_molecule_database_from_excel:13,test_subsample_molecule_database_from_pdb_dir:13,test_subsample_molecule_database_from_textfil:13,test_task_manag:13,test_taskmanag:5,test_topological_fprint_min_path_lesser_than_atom:13,test_tsne_transform:13,test_vectornorm_length_error:13,test_verbose_output:13,testcas:13,testcomparetargetmolecul:13,testdescriptor:13,testloadingerrorexcept:13,testmeasuresearch:13,testmolecul:13,testmoleculeset:13,testmultithread:13,testsimilaritymeasur:13,testsimilaritymeasurevalueerror:13,testtaskmanag:13,text:[0,7,13],textfil:13,than:[0,10],theori:0,therebi:0,therefor:8,thi:[0,7,8,9,10,13],third:10,though:13,thread:13,three:0,through:[0,13],thu:0,ti:7,tibshirani:0,time:[0,13],titl:10,tkinter:7,to_key_val_str:10,to_numpi:8,to_rdkit:8,todeschini:1,todeschini_1:1,todeschini_2:1,todeschini_3:1,todeschini_4:1,todeschini_5:1,toler:0,too:0,tool:0,top:9,topolog:0,torsion:10,tort:0,train:0,transform:[0,13],transit:7,tsne:13,tupl:7,tutori:2,twine:0,two:[0,7,8,13],type:[6,9,10,13],typeerror:13,typic:7,ui:[3,5,8],uintsparseintvec:8,un:[0,7],under:0,unimpl:13,uniniti:7,uniqu:8,unit:13,unittest:[0,13],unseen:0,unsupervis:13,up:13,upload:0,urbani:1,us:[0,6,7,8,9,10,13],user:[0,8],util:[5,6],v2:0,valu:[0,7,8,9,10,13],valueerror:[6,7,13],variat:0,vector:[7,8,10,13],verbos:13,verif:0,verifi:[0,13],version:0,via:[0,9],view:0,virtual:0,visual:[0,9],visualize_dataset:[5,6],visualizedataset:9,vlacho:0,vlachosgroup:0,vs:10,w:13,ward:0,warranti:0,watson:0,we:0,welcom:0,well:0,when:[0,6,9,13],where:0,whether:0,which:[0,7,8,9,10,13],whom:0,why:0,widm:0,willi:0,window:7,wire:0,without:[0,6],word:7,work:[2,13],x:[7,8,10],xlabel:10,xlabel_fonts:10,xtick:10,xtick_label:10,y:[0,10],yaml:0,ye:1,yield:0,ylabel:10,ylabel_fonts:10,your:0,yule:1,yule_1:1,yule_2:1},titles:["molSim README","Supported Similarity Metrics","molSim documentation","interfaces package","interfaces.UI package","molSim","molSim package","molSim.chemical_datastructures package","molSim.ops package","molSim.tasks package","molSim.utils package","setup module","test module","tests package"],titleterms:{"function":0,ccbmlib_fingerprint:10,chemical_datastructur:7,cite:0,cluster:8,cluster_data:9,compare_target_molecul:9,config_read:3,content:[2,3,4,6,7,8,9,10,13],contributor:0,current:0,descriptor:8,develop:0,document:[0,2],except:6,fingerprint:0,identify_outli:9,implement:0,indic:2,instal:0,interfac:[3,4],licens:0,measure_search:9,metric:1,modul:[3,4,6,7,8,9,10,11,12,13],molecul:7,molecule_set:7,molsim:[0,2,5,6,7,8,9,10],molsim_ui_main:4,note:0,op:8,output:8,packag:[3,4,6,7,8,9,10,13],plotting_script:10,purpos:0,readm:0,run:0,score:0,see_property_variation_with_similar:9,setup:11,similar:[0,1],similarity_measur:8,submodul:[3,4,6,7,8,9,10,13],subpackag:[3,6],support:[1,8],tabl:2,task:9,task_manag:9,test:[12,13],test_comparetargetmolecul:13,test_descriptor:13,test_loadingerrorexcept:13,test_measuresearch:13,test_molecul:13,test_moleculeset:13,test_multithread:13,test_similaritymeasur:13,test_similaritymeasurevalueerror:13,test_taskmanag:13,tutori:0,type:8,ui:4,util:10,visualize_dataset:9,work:0}})