import random
import struct
import logging
import json
import sys, os
import pandas as pd

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, 'pytorchfi')
import torch
import pytorchfi
import numpy as np
import pytorchfi.core as core
from pytorchfi.util import random_value
from pytorchfi.core import FaultInjection

logger=logging.getLogger("Fault_injection") 
logger.setLevel(logging.DEBUG) 

# _bf_inj_w_mask=0
# _layer=0


# def float_to_hex(f):
#     h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
#     return h[2:len(h)]


# def hex_to_float(h):
#     return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

# def int_to_float(h):
#     return float(struct.unpack(">f",struct.pack(">I",h))[0])
    


# def _log_faults(Log_string):
#     with open("./FSIM_logs/Fsim_log.log",'a+') as logfile:
#         logfile.write(Log_string+'\n')

# def bit_flip_weight_inj(pfi: core.FaultInjection, layer, k, c_in, kH, kW, inj_mask):
#     global _bf_inj_w_mask
#     global _layer
#     _bf_inj_w_mask=inj_mask
#     _layer=layer
#     return pfi.declare_weight_fault_injection(
#         function=_bit_flip_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW
#     )

# def _bit_flip_weight(data, location):
#     orig_data=data[location].item()
#     data_32bit=int(float_to_hex(data[location].item()),16)

#     #print(_bf_inj_w_mask)

#     corrupt_32bit=data_32bit ^ int(_bf_inj_w_mask[0])
#     corrupt_val=int_to_float(corrupt_32bit)
#     #print(data_32bit,_bf_inj_w_mask,corrupt_32bit, orig_data, corrupt_val)
#     log_msg=f"F_descriptor: Layer:{_layer}, (K, C, H, W):{location}, BitMask:{_bf_inj_w_mask[0]}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"
#     logger.info(log_msg)
#     _log_faults(log_msg)
#     return corrupt_val

#     # k, c, H, W

        
        
    # def update_fault_dictionary(self,key,val):           
    #     if self._layer not in self.fault_dictionary:
    #         self.fault_dictionary[self._layer]={}            
    #     if self._kK not in self.fault_dictionary[self._layer]:
    #         self.fault_dictionary[self._layer][self._kK]={}
    #     if self._kC not in  self.fault_dictionary[self._layer][self._kK]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC]={}
    #     if self._kH not in  self.fault_dictionary[self._layer][self._kK][self._kC]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH]={}
    #     if self._kW not in  self.fault_dictionary[self._layer][self._kK][self._kC][self._kH]:                            
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW]={}  
    #     if self._inj_mask not in  self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW]:
    #         self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW][self._inj_mask]={}
    #     self.fault_dictionary[self._layer][self._kK][self._kC][self._kH][self._kW][self._inj_mask][key]=val

        # k, c, H, W

def generate_fault_list_sbfm(path,pfi_model:FaultInjection, **kwargs):
    T=1.64485362695147  #confidence level
    E=0.01              #error margin
    P=0.5               # Perrror
    MSB_inection=30
    LSB_injection=28
    fault_list=[]   
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            f_list=pd.DataFrame(columns=['layer','kernel','channel','row','col','bitmask'])            
            layer_param=kwargs.get('layer')
            kK_param=kwargs.get('kernel')
            kC_param=kwargs.get('channel')
            pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())
            if layer_param!= None:
                layr=layer_param
            else:
                layr=random.randint(0,pfi_model.get_total_layers()-1)
            weight_shape=list(pfi_model.get_weights_size(layr))

            if kK_param != None:
                N=1
            else:
                N=weight_shape[0]
            
            if kC_param != None:
                N*=1
            else:
                N*=weight_shape[1]
            N=N*weight_shape[2]*weight_shape[3]*(MSB_inection-LSB_injection+1)

            #print(N)
            n=int(N/(1+(E**2)*(N-1)/((T**2)*P*(1-P))))
            #print(n)                
            i=0
            while i<n:
                if kK_param != None:
                    k=kK_param
                else:
                    k=random.randint(0,weight_shape[0]-1)                
                if kC_param != None:
                    c=kC_param
                else:
                    c=random.randint(0,weight_shape[1]-1)
                h=random.randint(0,weight_shape[2]-1)
                w=random.randint(0,weight_shape[3]-1)
                mask=2**(random.randint(19,31))                
                fault=[layr,k,c,h,w,mask]
                if fault not in fault_list :
                    fault_list.append(fault)
                    fault_dict={'layer':layr,'kernel':k,'channel':c,'row':h,'col':w,'bitmask':mask}
                    new_row=pd.DataFrame(fault_dict, index=[0])
                    f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)                                                        
                    i+=1
            f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list.values.tolist())


def generate_fault_list_ber(path,pfi_model:FaultInjection, **kwargs):
    fault_list=[]   
    fault_dict={}
    f_list=pd.DataFrame()             
    if kwargs:                               
        fault_list_file=kwargs.get('f_list_file')
        if not os.path.exists(os.path.join(path,fault_list_file)):                  
            # f_list=pd.DataFrame(columns=['layer','kernel','channel','height','width','bitmask'])     
            BER=kwargs.get('BER')   
            Num_trials=kwargs.get('trials')     
            pfi_model.print_pytorchfi_layer_summary()            
            print(pfi_model.get_total_layers())

            # fault_dict['ber']=kwargs.get('BER')

            if kwargs.get('layer')!= None:
                fault_dict['layer']=kwargs.get('layer')
            if kwargs.get('kernel') != None:
                fault_dict['kernel']=kwargs.get('kernel')
            if kwargs.get('channel')!=None:
                fault_dict['channel']=kwargs.get('channel')
            if kwargs.get('row')!=None:
                fault_dict['row']=kwargs.get('row')
            if kwargs.get('col')!=None:
                fault_dict['col']=kwargs.get('col')

            for i in (range(Num_trials)):
                for brate in range(0,BER+1):
                    fault_dict['ber']=brate
                    new_row=pd.DataFrame(fault_dict, index=[0])
                    f_list=pd.concat([f_list, new_row],ignore_index=True, sort=False)                                                        
                    i+=1
                f_list.to_csv(os.path.join(path,fault_list_file),sep=',')
        else:
            f_list = pd.read_csv(os.path.join(path,fault_list_file),index_col=[0]) 
    return(f_list.values.tolist())


class FI_report(object):
    def __init__(self,log_pah) -> None:
        self._k=0
        self._c_in=0
        self._kH=0
        self._kW=0
        self._layer=0    
        self._num_images=0
        self._gold_acc1=0
        self._gold_acck=0
        self._faul_acc1=0
        self._faul_acck=0
        
        self._golden_dictionary={}
        self._FI_results_dictionary={}
        self._fault_dictionary={}        
        self.Top1_faulty_code=0
        self.Topk_faulty_code=0
        self.T1_SDC=0
        self.T1_Masked=0
        self.T1_Critical=0

        self.T5_SDC=0
        self.T5_Masked=0
        self.T5_Critical=0

        self.Golden={}
        self.SDC_top1=0
        self.SDC_top5=0
        self.Critical_top1=0
        self.Critical_top5=0
        self.Masked_top1=0
        self.Masked_top5=0
        self.Gacc1=0
        self.Gacc5=0
        self.Facc1=0
        self.Facc5=0

        self.Accm_SDC_top1=0
        self.Accm_SDC_top5=0
        self.Accm_Critical_top1=0
        self.Accm_Critical_top5=0
        self.Accm_Masked_top1=0
        self.Accm_Masked_top5=0
        self.log_path=log_pah
        self.report_summary={}
        self._fsim_report=pd.DataFrame()
        self.check_point={
            "fault_idx":0,
            "top1": {
                "fault":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        },
                "images":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        }
            },
            "topk": {
                "fault":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        },
                "images":{
                        "Critical":0,
                        "SDC":0,
                        "Masked":0
                        }
            }
        }

    def set_fault_report(self, data):
        for key in data:
            self._fault_dictionary[key]=data[key]

    def load_check_point(self,chpt_file):
        ckpt_path_file=os.path.join(self.log_path,chpt_file)
        if not os.path.exists(ckpt_path_file):
            with open(ckpt_path_file,'w') as Golden_file:
                json.dump(self.check_point,Golden_file)
        else:
            with open(ckpt_path_file,'r') as Golden_file:
                chpt=json.load(Golden_file)
                self.check_point=chpt

        self.load_golden_report()

        if not os.path.exists(os.path.join(self.log_path,'fsim_report.csv')):
            self._fsim_report=pd.DataFrame(columns=['gold_ACC@1','gold_ACC@k',
                                        'img_Top1_Crit','img_Top1_SDC','img_Top1_Masked',
                                        'img_Topk_Crit','img_Topk_SDC','img_Topk_Masked',
                                        'fault_ACC@1','fault_ACC@k','Class_Top1','Class_Topk'])  
            self._fsim_report.to_csv(os.path.join(self.log_path,'fsim_report.csv'),sep=',')
        else:
            self._fsim_report = pd.read_csv(os.path.join(self.log_path,'fsim_report.csv'),index_col=[0])           
        

    def update_check_point(self,chpt_file):  
        self.__Update_counters()    
        fidx=self.check_point["fault_idx"]
        Report_name=f"FI_{fidx}_results.json"
        FI_report_json_file=os.path.join(self.log_path,Report_name)
        with open(FI_report_json_file,'w') as Golden_file:
            json.dump(self._FI_results_dictionary,Golden_file)        
        self._FI_results_dictionary={}

        new_row=pd.DataFrame(self._fault_dictionary, index=[0])
        self._fsim_report=pd.concat([self._fsim_report, new_row],ignore_index=True, sort=False)
        self._fsim_report.to_csv(os.path.join(self.log_path,'fsim_report.csv'),sep=',')  
              
        ckpt_path_file=os.path.join(self.log_path,chpt_file)
        with open(ckpt_path_file,'w') as Golden_file:
            json.dump(self.check_point,Golden_file)

    def __Update_counters(self):    
        self.Top1_faulty_code=0 # 0: Masked; 1: SDC; 2; Critical
        self.Topk_faulty_code=0 # 0: Masked; 1: SDC; 2; Critical
        self.check_point["fault_idx"]+=1 
        self.check_point["top1"]["images"]["Critical"]+=self.T1_Critical
        self.check_point["top1"]["images"]["SDC"]+=self.T1_SDC
        self.check_point["top1"]["images"]["Masked"]+=self.T1_Masked
        self.check_point["topk"]["images"]["Critical"]+=self.T5_Critical
        self.check_point["topk"]["images"]["SDC"]+=self.T5_SDC
        self.check_point["topk"]["images"]["Masked"]+=self.T5_Masked

            # break
        if(self.T1_Critical!=0):
            # self.Critical_top1+=1
            self.check_point["top1"]["fault"]["Critical"]+=1
            self.Top1_faulty_code=2
        elif self.T1_SDC !=0:
            # self.SDC_top1+=1
            self.check_point["top1"]["fault"]["SDC"]+=1
            self.Top1_faulty_code=1
        else:
            # self.Masked_top1+=1
            self.check_point["top1"]["fault"]["Masked"]+=1
            self.Top1_faulty_code=0

        if(self.T5_Critical!=0):
            # self.Critical_top5+=1
            self.check_point["topk"]["fault"]["Critical"]+=1
            self.Topk_faulty_code=2
        elif self.T5_SDC !=0:
            # self.SDC_top5+=1
            self.check_point["topk"]["fault"]["SDC"]+=1
            self.Topk_faulty_code=1
        else:
            # self.Masked_top5+=1  
            self.check_point["topk"]["fault"]["Masked"]+=1
            self.Topk_faulty_code=0

        GACC1=self._gold_acc1*100/self._num_images
        GACCk=self._gold_acck*100/self._num_images
        FACC1=self._faul_acc1*100/self._num_images
        FACCk=self._faul_acck*100/self._num_images


        self._fault_dictionary['gold_ACC@1'] = GACC1.item()
        self._fault_dictionary['gold_ACC@k'] = GACCk.item()

        self._fault_dictionary['img_Top1_Crit'] = self.T1_Critical
        self._fault_dictionary['img_Top1_SDC'] = self.T1_SDC
        self._fault_dictionary['img_Top1_Masked'] = self.T1_Masked
        self._fault_dictionary['fault_ACC@1'] = FACC1.item()

        self._fault_dictionary['img_Topk_Crit'] = self.T5_Critical
        self._fault_dictionary['img_Topk_SDC'] = self.T5_SDC
        self._fault_dictionary['img_Topk_Masked'] = self.T5_Masked
        self._fault_dictionary['fault_ACC@k'] = FACCk.item()
        
        self._fault_dictionary['Class_Top1'] = self.Top1_faulty_code
        self._fault_dictionary['Class_Topk'] = self.Topk_faulty_code


        self.T1_SDC=0
        self.T1_Masked=0
        self.T1_Critical=0
        self.T5_SDC=0
        self.T5_Masked=0
        self.T5_Critical=0
        self._gold_acc1=0
        self._gold_acck=0
        self._faul_acc1=0
        self._faul_acck=0
        self._num_images=0    

    def create_golden_report(self):
        listdirectories=os.listdir(self.log_path)
        num=0    
        for dir in listdirectories:      
            if "Golden_results_" in dir:            
                num+=1
        num+=1
        new_golden_name=f"Golden_results_{num}.json"
        os.system(f"mv {os.path.join(self.log_path,'Golden_results.json')} {os.path.join(self.log_path,new_golden_name)}")
        self._golden_dictionary={}
        self.load_golden_report()

    def load_golden_report(self):
        golden_file_path=os.path.join(self.log_path,'Golden_results.json')
        if not os.path.exists(golden_file_path):
            with open(golden_file_path,'w') as Golden_file:
                json.dump(self._golden_dictionary,Golden_file)
        else:
            
            with open(golden_file_path,'r') as Golden_file:
                self._golden_dictionary=json.load(Golden_file)

    def save_golden_report(self):        
        golden_file_path=os.path.join(self.log_path,'Golden_results.json')
        with open(golden_file_path,'w') as Golden_file:
            json.dump(self._golden_dictionary,Golden_file)

    def update_golden_report(self,index,output,target,topk=(1,)):
        maxk=max(topk)
        self._golden_dictionary[index]={}
        pred, clas=output.cpu().topk(maxk,1,True,True)
        self._golden_dictionary[index]['pred']=pred.tolist()
        self._golden_dictionary[index]['clas']=clas.tolist()
        self._golden_dictionary[index]['target']=target.cpu().tolist()

        
    def check_FI_report(self,index,output,target,topk=(1,)):        
        ResTop1=""
        ResTop5=""
        self.Golden=self._golden_dictionary[str(index)]
        FI_results_json={}        
        G_pred=torch.tensor(self.Golden['pred'],requires_grad=False).t()
        G_clas=torch.tensor(self.Golden['clas'],requires_grad=False).t()
        G_target=torch.tensor(self.Golden['target'],requires_grad=False)
        batch_size = G_target.size(0)

        maxk=max(topk)
        mink=min(topk)
        pred, clas=output.cpu().topk(maxk,1,True,True)
        FI_results_json['pred']=pred.tolist()
        FI_results_json['clas']=clas.tolist()
        FI_results_json['target']=target.cpu().tolist()
        
        FI_pred=pred.clone().detach().t()
        FI_clas=clas.clone().detach().t()
        FI_target=target.cpu().clone().detach()

        CMPGolden=G_clas.eq(G_target[None])
        CMPFaulty=FI_clas.eq(FI_target[None]) 

        self.Gacc1=CMPGolden[:mink].sum(dim=0,dtype=torch.float32)
        self.Facc1=CMPFaulty[:mink].sum(dim=0,dtype=torch.float32)
        self.Gacc5=CMPGolden[:maxk].sum(dim=0,dtype=torch.float32)
        self.Facc5=CMPFaulty[:maxk].sum(dim=0,dtype=torch.float32)

        CMPpredGoldFaulty=G_pred.eq(FI_pred).sum(dim=0,dtype=torch.float32)
        CMPClasGoldFaulty=G_clas.eq(FI_clas).sum(dim=0,dtype=torch.float32)
        ResTop1=[]
        ResTop5=[]

        for img in range(batch_size):                
            if self.Gacc1[img] == self.Facc1[img]:
                if(CMPpredGoldFaulty[img] == CMPClasGoldFaulty[img]):
                    self.T1_Masked+=1                
                    ResTop1.append("Masked")
                    # print(CMPpredGoldFaulty)
                    # print(CMPClasGoldFaulty)
                else:
                    self.T1_SDC+=1
                    ResTop1.append("SDC")
            else:
                self.T1_Critical+=1
                ResTop1.append("Critical")

            if self.Gacc5[img] == self.Facc5[img]:
                if(CMPpredGoldFaulty[img] == CMPClasGoldFaulty[img]):
                    self.T5_Masked+=1
                    ResTop5.append("Masked")
                    # print(CMPpredGoldFaulty)
                    # print(CMPClasGoldFaulty)
                else:
                    self.T5_SDC+=1
                    ResTop5.append("SDC")
            else:
                self.T5_Critical+=1
                ResTop5.append("Critical")
            
        FI_results_json['ResTop1']=ResTop1
        FI_results_json['ResTopk']=ResTop5

        gold_result_list = []
        faul_result_list = []
        for k in topk:
            gold_correct_k = CMPGolden[:k].flatten().sum(dtype=torch.float32)
            faul_correct_k = CMPFaulty[:k].flatten().sum(dtype=torch.float32)
            gold_result_list.append(gold_correct_k) 
            faul_result_list.append(faul_correct_k) 

        self._num_images+=batch_size
        self._faul_acc1+=faul_result_list[0]
        self._faul_acck+=faul_result_list[1]

        self._gold_acc1+=gold_result_list[0]
        self._gold_acck+=gold_result_list[1]

        self._FI_results_dictionary[index]={}
        self._FI_results_dictionary[index]=FI_results_json

        return(FI_results_json)


    def FI_results(self):
        return (self.check_point["top1"]["fault"]["Critical"], self.check_point["top1"]["fault"]["SDC"], self.check_point["top1"]["fault"]["Masked"], 
        self.check_point["topk"]["fault"]["Critical"], self.check_point["topk"]["fault"]["SDC"], self.check_point["topk"]["fault"]["Masked"])

    def FI_results_accm(self):
        return (self.check_point["top1"]["images"]["Critical"], self.check_point["top1"]["images"]["SDC"], self.check_point["top1"]["images"]["Masked"], 
        self.check_point["topk"]["images"]["Critical"], self.check_point["topk"]["images"]["SDC"], self.check_point["topk"]["images"]["Masked"])


class FI_framework(object):
    def __init__(self,log_path,mode='single_fault') -> None:
        # self.fault_dictionary={}   
        self._BER=0 
        self._layer=[]
        self._kK=[]
        self._kC=[]
        self._kH=[]
        self._kW=[]                
        self._inj_mask=[]     
        self._fault_list=[]    
        self._bf_inj_w_mask=0
        self.faultfreedata=0
        self.faultydata=0
        self.log_msg=''
        self.FI_report=FI_report(log_path)
        self.log_path=log_path
        self.log_file=os.path.join(log_path,'FSIM_log.log')
        self.pfi_model=None
        self.faulty_model=None
        try:
            os.mkdir(self.log_path)           
        except:
            print(f"The log path: {self.log_path} // already exist...")           


    def float_to_hex(self,f):
        h=hex(struct.unpack('<I', struct.pack('<f', f))[0])
        return h[2:len(h)]

    def hex_to_float(self,h):
        return float(struct.unpack(">f",struct.pack(">I",int(h,16)))[0])

    def int_to_float(self,h):
        return float(struct.unpack(">f",struct.pack(">I",h))[0])

    def log_faults(self,chpt_file):
        self.FI_report.update_check_point(chpt_file)                 
        with open(self.log_file,'a+') as logfile:
            self.log_msg
            logfile.write(self.log_msg+'\n')

    def create_fault_injection_model(self,device,model,batch_size=1,input_shape=[3,224,224],layer_types=[torch.nn.Conv2d]): 
        if device.type.startswith('cuda'): 
            use_cuda=True
        else:
            use_cuda=False
        self.pfi_model = FaultInjection(model, 
                    batch_size=batch_size,
                    input_shape=input_shape,
                    layer_types=layer_types,
                    use_cuda=use_cuda,
                    )

    def generate_fault_list(self, **kwargs):        
        if kwargs:
            if (kwargs.get('flist_mode')=='sbfm'):
                self._fault_list=generate_fault_list_sbfm(self.log_path,self.pfi_model,**kwargs)
            elif(kwargs.get('flist_mode')=='ber'):
                self._fault_list=generate_fault_list_ber(self.log_path,self.pfi_model,**kwargs)
            else:
                raise ValueError("The fault list can't be generated in this configuration")
        else:
            raise ValueError("The input arguments are wrong, please be sure you selected at least the fault list name in csv format")
        
    def iter_fault_list(self):
        for k in range(int(self.FI_report.check_point["fault_idx"]),len(self._fault_list)):
            fault=self._fault_list[k]
            yield fault            

    def Model_inject_fault(self,input):
        return(self.faulty_model(input))
       
    def bit_flip_weight_inj(self, layer, k, c_in, kH, kW, inj_mask):
        self._kK=(k)
        self._kC=(c_in)
        self._kH=(kH)
        self._kW=(kW)  
        self._inj_mask=(inj_mask)
        self._layer=layer
        self.faulty_model=self.pfi_model.declare_weight_fault_injection(
            BitFlip=self._bit_flip_weight, layer_num=layer, k=k, dim1=c_in, dim2=kH, dim3=kW, bitmask=inj_mask
        )
        self.faulty_model.eval()

    def _bit_flip_weight(self,data, location, injmask):
        orig_data=data[location].item()
        data_32bit=int(self.float_to_hex(data[location].item()),16)
        #print(f"{data.dtype}")
        #print(_bf_inj_w_mask)

        corrupt_32bit=data_32bit ^ int(injmask)
        corrupt_val=self.int_to_float(corrupt_32bit)
        #print(data_32bit,_bf_inj_w_mask,corrupt_32bit, orig_data, corrupt_val)
        self.log_msg=f"F_descriptor: Layer:{self._layer}, (K, C, H, W):{location}, BitMask:{injmask}, Ffree_Weight:{data_32bit}, Faulty_weight:{corrupt_32bit}"
        logger.info(self.log_msg)
        fsim_dict={'Layer':self._layer[0], 
                    'kernel':self._kK[0],
                    'channel':self._kC[0],
                    'row':self._kH[0],
                    'col':self._kW[0],
                    'BitMask':self._inj_mask[0],
                    'Ffree_Weight':data_32bit,
                    'Faulty_weight':corrupt_32bit,
                    'Abs_error':(orig_data-corrupt_val)}
        self.FI_report.set_fault_report(fsim_dict)     
        return corrupt_val
    

    def BER_weight_inj(self, BER, layer=None, kK=None, kC=None, kH=None, kW=None, inj_mask=None):       
        self._layer=[]
        self._kK=[]
        self._kC=[]
        self._kH=[]
        self._kW=[]                
        self._inj_mask=[]
        N_layers=1
        N_Kernels=1
        N_Channels=1
        N_Rows=1
        N_Cols=1
        N_Bits=32              
        Tot_N_bits=0
        Bit_mask_selected=1
        fsim_dict={}
        err_list=[]
        num_errors=0
        while(num_errors<BER):
            if layer:
                N_layers=1
                layer_selected=layer-1
                fsim_dict['layer']=layer_selected
            else:
                N_layers=self.pfi_model.get_total_layers()
                layer_selected=random.randint(0,N_layers-1)                    
            weight_shape=list(self.pfi_model.get_weights_size(layer_selected))
            if kK:
                N_Kernels=1                
                Kernel_selected=kK-1
                fsim_dict['kernel']=Kernel_selected
            else:
                N_Kernels=weight_shape[0]
                Kernel_selected=random.randint(0,N_Kernels-1)

            if kC:
                N_Channels=1                
                Channel_selected=kC-1
                fsim_dict['channel']=Channel_selected
            else:
                N_Channels=weight_shape[1]
                Channel_selected=random.randint(0,N_Channels-1)
            if kH:
                N_Rows=1                
                Row_selected=kH-1
                fsim_dict['row']=Row_selected
            else:
                N_Rows=weight_shape[2]
                Row_selected=random.randint(0,N_Rows-1)
            if kW:
                N_Cols=1                
                Col_selected=kH-1
                fsim_dict['col']=Col_selected
            else:
                N_Cols=weight_shape[3]
                Col_selected=random.randint(0,N_Cols-1)

            Bit_mask_selected=2**(random.randint(0,31))

            tmp=[layer_selected,Kernel_selected,Channel_selected,Row_selected,Col_selected,Bit_mask_selected]
            if tmp not in err_list:
                err_list.append(tmp)
                self._layer.append(layer_selected)
                self._kK.append(Kernel_selected)
                self._kC.append(Channel_selected)
                self._kH.append(Row_selected)
                self._kW.append(Col_selected)
                self._inj_mask.append(Bit_mask_selected)
                num_errors+=1                    

        Tot_N_bits=N_layers*N_Kernels*N_Channels*N_Rows*N_Cols*N_Bits
        self._BER=BER/Tot_N_bits
        fsim_dict['N_BER']=BER
        fsim_dict['T_Bits']=Tot_N_bits
        fsim_dict['BER']=self._BER
        self.FI_report.set_fault_report(fsim_dict)

        self.faulty_model=self.pfi_model.declare_weight_fault_injection(
            BitFlip=self._BER_weight, layer_num=self._layer, k=self._kK, dim1=self._kC, dim2=self._kH, dim3=self._kW, bitmask=self._inj_mask
        )
        self.faulty_model.eval()


    def _BER_weight(self,data, location, injmask):
        orig_data=data[location].item()
        data_32bit=int(self.float_to_hex(data[location].item()),16)
        corrupt_32bit=data_32bit ^ int(injmask)
        corrupt_val=self.int_to_float(corrupt_32bit)

        return corrupt_val




class DatasetSampling(object):
    def __init__(self,dataset,num_images):        
        self.length=len(dataset)
        self.num_images=num_images
        self.indices=[]

    def listindex(self):
        self.indices=[]
        for i in range(0,self.length):
            if(i%50)<self.num_images:
                self.indices.append(i)
        return(self.indices)

    def __len__(self):
        return len(self.indices)