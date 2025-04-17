library(CGWAS)


input_data_dir <- '/home/wcy/data/UKB/eye_feature/cgwas/right_vessel_dim128/' 

outputPath <- '/home/wcy/data/UKB/eye_feature/cgwas/right_vessel_dim128/'

#outputPath <- '/home/wcy/data/UKB/eye_feature/test/'


gwasFileName <- c("latent_1.assoc", "latent_10.assoc", "latent_100.assoc", "latent_101.assoc", "latent_102.assoc", "latent_103.assoc", "latent_104.assoc", "latent_105.assoc", "latent_106.assoc", "latent_107.assoc", "latent_108.assoc", "latent_109.assoc", "latent_11.assoc", "latent_110.assoc", "latent_111.assoc", "latent_112.assoc", "latent_113.assoc", "latent_114.assoc", "latent_115.assoc", "latent_116.assoc", "latent_117.assoc", "latent_118.assoc", "latent_119.assoc", "latent_12.assoc", "latent_120.assoc", "latent_121.assoc", "latent_122.assoc", "latent_123.assoc", "latent_124.assoc", "latent_125.assoc", "latent_126.assoc", "latent_127.assoc", "latent_128.assoc", "latent_13.assoc", "latent_14.assoc", "latent_15.assoc", "latent_16.assoc", "latent_17.assoc", "latent_18.assoc", "latent_19.assoc", "latent_2.assoc", "latent_20.assoc", "latent_21.assoc", "latent_22.assoc", "latent_23.assoc", "latent_24.assoc", "latent_25.assoc", "latent_26.assoc", "latent_27.assoc", "latent_28.assoc", "latent_29.assoc", "latent_3.assoc", "latent_30.assoc", "latent_31.assoc", "latent_32.assoc", "latent_33.assoc", "latent_34.assoc", "latent_35.assoc", "latent_36.assoc", "latent_37.assoc", "latent_38.assoc", "latent_39.assoc", "latent_4.assoc", "latent_40.assoc", "latent_41.assoc", "latent_42.assoc", "latent_43.assoc", "latent_44.assoc", "latent_45.assoc", "latent_46.assoc", "latent_47.assoc", "latent_48.assoc", "latent_49.assoc", "latent_5.assoc", "latent_50.assoc", "latent_51.assoc", "latent_52.assoc", "latent_53.assoc", "latent_54.assoc", "latent_55.assoc", "latent_56.assoc", "latent_57.assoc", "latent_58.assoc", "latent_59.assoc", "latent_6.assoc", "latent_60.assoc", "latent_61.assoc", "latent_62.assoc", "latent_63.assoc", "latent_64.assoc", "latent_65.assoc", "latent_66.assoc", "latent_67.assoc", "latent_68.assoc", "latent_69.assoc", "latent_7.assoc", "latent_70.assoc", "latent_71.assoc", "latent_72.assoc", "latent_73.assoc", "latent_74.assoc", "latent_75.assoc", "latent_76.assoc", "latent_77.assoc", "latent_78.assoc", "latent_79.assoc", "latent_8.assoc", "latent_80.assoc", "latent_81.assoc", "latent_82.assoc", "latent_83.assoc", "latent_84.assoc", "latent_85.assoc", "latent_87.assoc", "latent_88.assoc", "latent_89.assoc", "latent_9.assoc", "latent_90.assoc", "latent_91.assoc", "latent_92.assoc", "latent_93.assoc", "latent_94.assoc", "latent_95.assoc", "latent_96.assoc", "latent_97.assoc", "latent_98.assoc", "latent_99.assoc")

traitName <- c("latent_1", "latent_10", "latent_100", "latent_101", "latent_102", "latent_103", "latent_104", "latent_105", "latent_106", "latent_107", "latent_108", "latent_109", "latent_11", "latent_110", "latent_111", "latent_112", "latent_113", "latent_114", "latent_115", "latent_116", "latent_117", "latent_118", "latent_119", "latent_12", "latent_120", "latent_121", "latent_122", "latent_123", "latent_124", "latent_125", "latent_126", "latent_127", "latent_128", "latent_13", "latent_14", "latent_15", "latent_16", "latent_17", "latent_18", "latent_19", "latent_2", "latent_20", "latent_21", "latent_22", "latent_23", "latent_24", "latent_25", "latent_26", "latent_27", "latent_28", "latent_29", "latent_3", "latent_30", "latent_31", "latent_32", "latent_33", "latent_34", "latent_35", "latent_36", "latent_37", "latent_38", "latent_39", "latent_4", "latent_40", "latent_41", "latent_42", "latent_43", "latent_44", "latent_45", "latent_46", "latent_47", "latent_48", "latent_49", "latent_5", "latent_50", "latent_51", "latent_52", "latent_53", "latent_54", "latent_55", "latent_56", "latent_57", "latent_58", "latent_59", "latent_6", "latent_60", "latent_61", "latent_62", "latent_63", "latent_64", "latent_65", "latent_66", "latent_67", "latent_68", "latent_69", "latent_7", "latent_70", "latent_71", "latent_72", "latent_73", "latent_74", "latent_75", "latent_76", "latent_77", "latent_78", "latent_79", "latent_8", "latent_80", "latent_81", "latent_82", "latent_83", "latent_84", "latent_85", "latent_87", "latent_88", "latent_89", "latent_9", "latent_90", "latent_91", "latent_92", "latent_93", "latent_94", "latent_95", "latent_96", "latent_97", "latent_98", "latent_99")


#gwasFileName <- c("latent_1.assoc", "latent_10.assoc", "latent_100.assoc")
#traitName <- c("latent_1", "latent_10", "latent_100")


gwasFilePath <- file.path(input_data_dir, gwasFileName)

snpFilePath <- file.path(input_data_dir, 'snp_list.csv')

mrafFilePath <- file.path(input_data_dir, 'maf_mean_values.csv')


cgwas(gwasFilePath, snpFilePath, outputPath,
      traitName = traitName, mrafFilePath = mrafFilePath, indSNPN = 1e5,threadN = 30)
