#! /sma/local/anaconda/bin/python
import numpy as np
import pandas as pd
import os, sys, struct, mmap, argparse, shutil
import pickle
from pprint import pprint as pp
import pickle as pickle
import matplotlib.pyplot as plt
import math as math






#need to import .csv files that are found in output directory of sma2matlab_1012.py (called test1 for now)

#unpack .csv files to managable format
input_path = "./"
files = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if ".csv" in fn]

# data is list of all tuples
data = [] #need to have to determine number of elements in data structure to create dictionary to stuff all info in for later
header = None
for fn in files:
    with open(fn, "r") as f:
        lines = f.readlines()
    if not header: #only need to write in header once 
        header = lines[0].strip().split(",") + ["datetime"]
    datetime = os.path.basename(fn).strip(".csv")
    # lines is all tuples for this file
    lines = [tuple(line.strip().split(",") + [datetime]) for line in lines[1:]]
    data += lines

#print(header)
#pp(data)


sigStruct = {}
n=len(data)
sigStruct['Avg_sigma_rx1'] = np.zeros(n,dtype=np.float)
sigStruct['Avg_sigma_rx2'] = np.zeros(n,dtype=np.float)
sigStruct['Gunn_LO_rx1']=np.zeros(n,dtype=np.float)
sigStruct['Gunn_LO_rx2']=np.zeros(n,dtype=np.float)
sigStruct['souName'] = np.zeros(n,dtype=object)
sigStruct['datetime']=np.zeros(n,dtype=object)

for i in range(n):
    n, s1, s2, g1, g2, d = data[i]
    sigStruct['Avg_sigma_rx1'][i] = s1
    sigStruct['Avg_sigma_rx2'][i] = s2
    sigStruct['Gunn_LO_rx1'][i] = g1
    sigStruct['Gunn_LO_rx2'][i] = g2
    sigStruct['souName'][i] = n
    sigStruct['datetime'][i] = d
    

#iterate over source
gainSources=['0004-476','0005+383','0006-063','0006+243','0010+109','0013+408','0014+612','0019+203','0019+260','0019+734','0022+002','0034+279','0050-094','0051-068','0057+303','0102+584','0106-405','0108+015','0112+227','0113+498','0115-014','0116-116','0118-216','0120-270','0121+118','0121+043','0132-169','0136+478','0137-245','0141-094','0149+059','0149+189','0152+221','0204+152','0204-170','0205+322','0210-510','0217+738','0217+017','0219+013','0222-346','0224+069','0228+673','0237+288','0238+166','0239+042','0241-082','0242+110','0242-215','0244+624','0246-468','0251+432','0259+425','0303+472','0309+104','0310+382','0313+413','0319+415','0325+469','0325+226','0329-239','0334-401','0336+323','0339-017','0340-213','0346+540','0348-278','0354+467','0359+509','0359+323','0401+042','0403-360','0405-131','0406-384','0410+769','0415+448','0416-209','0418+380','0422+023','0423-013','0423+418','0424-379','0424+006','0428+329','0428-379','0433+053','0440-435','0442-002','0449+113','0449+635','0453-281','0455-462','0457-234','0457+067','0501-019','0502+061','0502+136','0502+416','0505+049','0509+056','0510+180','0512+152','0522-364','0526-485','0527+035','0530+135','0532+075','0533+483','0538-440','0539+145','0539-286','0541-056','0541+474','0542+498','0555+398','0605+405','0607-085','0608-223','0609-157','0625+146','0629-199','0646+448','0648-307','0650-166','0710+475','0717+456','0721+713''0725+144','0725-009','0730-116','0733+503','0738+177','0739+016','0741+312','0747+766','0747-331','0748+240','0750+482','0750+125','0753+538','0757+099','0802+181','0804-278','0806-268','0808-078','0808+498','0811+017','0818+423','0823+223','0824+558','0824+392','0825+031','0826-225','0828-375','0830+241','0831+044','0836-202','0840+132','0841+708','0854+201','0902-142','0903+468','0909+013','0914+027','0920+446','0921+622','0925+003','0927+390','0927-205','0937+501','0943-083','0948+406','0956+252','0957+553','0958+474','0958+655','1008+063','1010+828','1014+230','1018+357','1033+608','1035-201','1037-295','1039-156','1041+061','1043+241','1044+809','1048-191','1048+717','1048+717','1051+213','1057-245','1058+812','1058+015','1102+279','1103+302','1103+220','1104+382','1107-448','1107+164','1111+199','1112-219','1118+125','1120-251','1120+143','1122+180','1127-189','1130-148','1130-148','1145-228','1146-289','1146+539','1146+399','1147-382','1153+809','1153+495','1159-224','1159+292','1203+480','1205-265','1207+279','1209-241','1215-175','1218-460','1221+282','1222+042','1224+213','1227+365','1229+020','1230+123','1239+075','1244+408','1246-075','1246-257','1248-199','1254+116','1256-057','1258-223','1305-105','1309+119','1310+323','1316-336','1325-430','1327+221','1329+319','1331+305','1337-129','1337-129','1357-154','1357+767','1408-078','1415+133','1416+347','1419+543','1419+383','1424-492','1427-421','1433-158','1439-155','1446+173','1454-377','1454-402','1458+042','1459+716','1504+104','1505+034','1506+426','1507-168','1510-057','1512-090','1513-102','1513-212','1516+002','1517-243','1522-275','1538+003','1540+147','1549+506','1549+026','1549+026','1550+054','1553+129','1557-000','1604-446','1608+104','1613+342','1617+027','1619+227','1625-254','1626-298','1632+825','1635+381','1637+472','1638+573','1640+397','1642+689','1642+398','1653+397','1658+347','1658+476','1658+076','1700-261','1707+018','1713-269','1716+686','1719+177','1727+455','1728+044','1733-130','1734+389','1734+094','1737+063','1739+499','1740+521','1743-038','1744-312','1751+096','1753+288','1800+388','1800+784','1801+440','1802-396','1806+698','1821+397','1824+568','1829+487','1830+063','1832-206','1842+681','1848+323','1848+327','1849+670','1902+319','1911-201','1923-210','1924+156','1924-292','1925+211','1927+612','1927+739','1937-399','1955+515','1957-387','2000-178','2005+778','2007+404','2009-485','2009+724','2011-157','2012+464','2015+371','2016+165','2023+318','2025+032','2025+337','mwc349a','2035+109','2038+513','2047-166','2049+100','2050+041','2056-472','2057-375','2101-295','2101+036','2109+355','2123+055','2129-183','2131-121','2134-018','2134-018','2138-246','2139+143','2142-046','2147+094','2148+069','2151+071','2151-304','2152+175','2158-150','2158-302','2202+422','2203+317','2203+174','2206-185','2213-254','2217+243','2218-035','2225-049','2229-085','2232+117','2235-485','2236+284','2243-257','2246-121','2247+031','2248-325','2253+161','2258-279','2301+374','2302-373','2307+148','2311+344','2320+052','2321+275','2323-032','2327+096','2329-475','2331-159','2333-237','2334+076','2337-025','2347+517','2347-512','2348-165','2354+458','2358-103',' 3c454.3','3c446',' bllac','p2134+0','3c418','3c395','3c380','3c371','nrao530','mrk501','3c345','nrao512','3c309.1','3c286','cen a','3c279',' 3c274','3c273','mrk421','oj287','3c207','3c147',' 3c120','3c111','3c84','ngc315','3C454.3','3C446',' bllac','p2134+0','3C418','3C395','3C380','3C371','nrao530','mrk501','3C345','nrao512','3C309.1','3C286','Cen a','3C279',' 3C274','mrk421','oj287','3C273','3C207','3C147','3C120','3C111','3C84']


calSig= {}
#calSig['cal_Sigma']=np.zeros(len(gainSources),dtype=np.float)
#calSig['souName']=np.zeros(len(gainSources),dtype=object)

#iterate over freq
#for each freq: 1) see if there's sigma data in structure 2) if data, then append list, 3) normalize sigma data per freq
##for each source, plot normalized sigma data per freq
for j in range(len(gainSources)):
    freq_bin=range(190,371)
    cal_sigma = [None]*len(freq_bin) #one list per source; consider creating structure to save this to for easy plotting later
    cal_sigma_3c84 = [None]*len(freq_bin)
    for i in range(len(freq_bin)):
        #mask for source name and freq match
        gunnLO_rx1 = sigStruct['Gunn_LO_rx1']
        select = np.where(np.logical_and(np.logical_or(abs(gunnLO_rx1-8-freq_bin[i])<4,abs(gunnLO_rx1+8-freq_bin[i])<4),sigStruct['souName']==gainSources[j]))
        if len(select[0]) > 0: 
            cal_sigma[i] = (sum([(x**-2) for x in sigStruct['Avg_sigma_rx1'][select]])**-0.5)*np.sqrt(8/(freq_bin[i]/1000.))
    
    if all(x is None for x in cal_sigma)==False:
        #calSig['cal_Sigma'[j]]=cal_sigma
        #calSig['souName'][j]=gainSources[j]
        calSig[gainSources[j]] = cal_sigma
             


        #SAVE calSig AS PICKLE FILE 
f=open('cal_sensitivity.p','w')
pickle.dump(calSig,f)
f.close()

file=open('cal_sensitivity_data.p','wb')
pickle.dump(calSig,file,-1)
file.close()
#to open
#f=open('cal_sensitivity.p','r')
#cal_dict=pickle.load(f)
#f.close()

###plotting section######



f=open('cal_sensitivity_data.p','rb')
cal_sigma=pickle.load(f) #dictionary with sensitivities for each frequency bin for each calibrator
f.close()

#cal_sig= np.array(list(cal_sigma.items()) #convert dictionary to array
                
N_freq=len(cal_sigma['3c84']) 
N_sources=len(cal_sigma.keys())
cal_sig_array =np.zeros([N_freq,N_sources]) 

for x,source in enumerate(cal_sigma.keys()):
    for y in range(N_freq):
        cal_sig_array[y,x]=cal_sigma[source][y]
        
    
cal_sig_array[np.where(cal_sig_array!=cal_sig_array)]=1000000.

sensitivity_threshold=np.geomspace(.1,100,31)

#binStruct= {}
#binStruct['sensitivity']=np.zeros(len(sensitivity_threshold))
#binStruct['bin_count']=np.zeros(len(sensitivity_threshold))
N_freq=len(cal_sigma['3c84'])
N_threshold=len(sensitivity_threshold)
binned_array = np.zeros(shape=(N_freq,N_threshold))

 #first loop is going thru each frequency bin    
for i in range(N_freq):
  
 
    #sigma_bin=[x for x in cal_sig_array if np.isnan(cal_sig_array)==False]
    #sigma_bin=np.where(np.isnan(cal_sig_array)==False)
    
    #sigma_bin=np.array(sigma_bin)
   # bin_count_per_threshold = []

    #second loop is going thru each sensitivity threshold
    for j in range(N_threshold):
        binned_array[i,j]=np.sum(cal_sig_array[i]*1500<=sensitivity_threshold[j]) #factor of 1500 to take into account 300 kn/s and 5 sigma detection


fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
plot = ax.imshow(np.transpose(np.log10(binned_array)),extent=[190,371,100,.1],aspect='auto')

fig.colorbar(plot)
ax.set_yticklabels([.1,.5,3,13,63,100])


#plt.yscale('log')
#extent=[horizontal_min,horizontal_max,vertical_min,vertical_max].

y_label=['{:.2f}'.format(x) for x in sensitivity_threshold]
print(y_label)


#plt.axes.set_yticklabels('',y_label)
#ax1.set_xticklabels
#plt.Axes.axes('Frequency [GHz]')
#plt.colorbar()
ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel('Sensitivity Threshold')
plt.title('Sensitivity Thresholds for Calibrator Fields')
plt.show()



freq_bin=np.array(range(190,371,1))
plt.contour(np.log10(sensitivity_threshold), freq_bin,binned_array,[1,10,100])
plt.title('Contour Plot of Sensitivity Thresholds for Calibrator Fields')
plt.show()
#.1 is 20%, .3 is factor of 2, .5 is pi, 1 is 10
