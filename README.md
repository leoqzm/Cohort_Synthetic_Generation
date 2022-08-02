# Cohort_Synthetic_Generation
## Requirements
Python 3.8 or above \
Pytorch 1.10.0 or above \
NVIDIA GPU for training

## Environment
conda
```
wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh
chmod +x mini.sh
bash ./mini.sh -b -f -p /usr/local
```
```
conda create --name nemo python==3.9
conda activate nemo
```
Pytorch
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Nemo_toolkit(Need to install before Apex)
```
sudo apt-get install libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```
Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5d8c8a8eedaf567d56f0762a45431baf9c0e800e
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" ./
```
Other Requirements:
```

```
Show nemo env in notebook kernal:
```
conda install ipykernel
ipython kernel install --user --name=nemo
```
# Run Cohort_Synthetic_Data_Generation_Megatron.ipynb
## 1.Data Cleaning and Formatting
Need to follow the 1.Data Cleaning and Formatting part in the notebook
### Difficulties
The input dataset \to do was first 290549 rows with  2204 columns, and we need to reduce the size of the columns. 
For this purpose we use Multioutput_Models.ipynb to calculate the first 100 most related features for the diabetes and hypertension for the dataset.
Also I add columns which related to times , we need to seperate the year month date with this columns and add them into the datas. The reason is date is also a important imformation for the patient
```
df1['admityear']=df1['admittime'].str[0:4]
df1['admitmonth']=df1['admittime'].str[5:7]
df1['admitday']=df1['admittime'].str[8:10]

df1['dischyear']=df1['dischtime'].str[0:4]
df1['dischmonth']=df1['dischtime'].str[5:7]
df1['dischday']=df1['dischtime'].str[8:10]

df1['chartyear']=df1['charttime_hr'].str[0:4]
df1['chartmonth']=df1['charttime_hr'].str[5:7]
df1['chartday']=df1['charttime_hr'].str[8:10]
```
The dataset becomes 29951 rows × 105 columns
And after get the trimmed dataset, part of the columns datatype need to be modified. For example columns related to condition era which needs to be integer were in float, so change the datatype with this code block:
```
df['cera_30Hypertension'] = df['cera_30Hypertension'].astype("int")
df['cera30_sum'] = df['cera30_sum'].astype("int")
df['cera_30Alzheimer'] = df['cera_30Alzheimer'].astype("int")
df['cera_30AtrialFib'] = df['cera_30AtrialFib'].astype("int")

df['cera_30Covid'] = df['cera_30Covid'].astype("int") 
df['cera_30Depression']=df['cera_30Depression'].astype("int")
df['cera_30Diabetes']=df['cera_30Diabetes'].astype("int")
df['cera_30Alzheimer']=df['cera_30Alzheimer'].astype("int")
df['cera_30Pneumonia']=df['cera_30Pneumonia'].astype("int")

df['drugs_cnt_uniq'] = df['drugs_cnt_uniq'].astype('int')
df['resp_open'] = df['resp_open'].astype('int')
df['Trazodone'] = df['Trazodone'].astype('int')
```


## 2. Tokenizer training
Need to follow the 2. Tokenizer training for produce the tokenization for the dataset, and you can set NUM_OF_CPUS for the cpus you use 
### difficulties for tokenization
At first, we use the settings of credit_card and the first model does not converge that well with 3 tokenizer per columns for integer and float type.
The time columns becomes very big or very small like 24433
So we change to 6 tokenizer per columns with the code block:
```

tab_structure = []
for c in columns:
    if c in float_columns:
        item = {
            "name": c,
            "code_type": "float",
            "args": {
                "code_len": 6,  # number of tokens used to code the column
                "base": 32,   # the positional base number. ie. it uses 32 tokens for one digit
                "fillall": True, # whether to use full base number for each token or derive it from the data.
                "hasnan": False, # can it handles nan or not
                "transform": "yeo-johnson" # can be ['yeo-johnson', 'quantile', 'robust'], check https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing 
            }
        }
    elif c in integer_columns:
        item = {
            "name": c,
            "code_type": "int",
            "args": {
                "code_len": 6,  # number of tokens used to code the column
                "base": 47,   # the positional base number. ie. it uses 32 tokens for one digit
                "fillall": True, # whether to use full base number for each token or derive it from the data.
                "hasnan": True, # can it handles nan or not
            }
        }
    else:
        item = {
            "name": c,
            "code_type": "category",
        }
    tab_structure.append(item)
print(tab_structure)
print(OmegaConf.to_yaml(tab_structure))
print(columns)

example_arrays = {}
for col in tab_structure:
    col_name = col['name']
    if col_name in category_columns:
        example_arrays[col_name] = [i.strip() for i in df[col_name].astype(str).unique()]
    else:
        example_arrays[col_name] = df[col_name].dropna().unique()
cc = ColumnCodes.get_column_codes(tab_structure, example_arrays)
print('each row uses', sum(cc.sizes)+ 1, 'tokens')
with open(CC_OUTPUT_P, 'wb') as handle:
    pickle.dump(cc, handle)
```

## 3.Model configuration
Need to set the GPUS and hyperparameter for the model in this session
For 4 GPUS:
```
NUM_LAYERS = 4
NUM_GPUS = 4
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
SEQ_LENGTH = 1024
TENSOR_MP_SIZE = 2
PIPELINE_MP_SIZE = 2
```
For 1 GPU:
```
NUM_LAYERS = 4
NUM_GPUS = 1
HIDDEN_SIZE = 512
NUM_ATTENTION_HEADS = 8
SEQ_LENGTH = 512
TENSOR_MP_SIZE = 1
PIPELINE_MP_SIZE = 1
```

## 4.Preprocess
IF the jupyternotebook code doesn't work(for example No module named 'ftfy') 
copy paste the follow codes into terminal to run preprocess
{NUM_OF_CPUS} is for the number of the CPUS, we use 16 cpus
```
python preprocess_data_for_megatron.py \
    --input=Cohort_data/HT_Pros_card.jn \
    --json-keys=text \
    --tokenizer-library=tabular \
    --vocab-file=Cohort_data/HT_Pros_coder.pickle \
    --tokenizer-type=Tabular \
    --output-prefix=Cohort_tabular_data \
    --delimiter=, \
    --workers={NUM_OF_CPUS}
```
## 5. Pretrain
```
python megatron_gpt_pretraining.py \
        trainer.devices={NUM_GPUS} \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=100 \
        trainer.val_check_interval=500 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=10000 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=Cohort_Latest_results \
        model.tensor_model_parallel_size={TENSOR_MP_SIZE} \
        model.pipeline_model_parallel_size={PIPELINE_MP_SIZE} \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings={SEQ_LENGTH} \
        model.encoder_seq_length={SEQ_LENGTH} \
        model.data.seq_length={SEQ_LENGTH} \
        model.tokenizer.type=Tabular \
        model.tokenizer.library=tabular \
        model.tokenizer.vocab_file={CC_OUTPUT_P} \
        model.tokenizer.delimiter=\',\' \
        model.data.eod_mask_loss=True \
        model.data.splits_string=\'3800,198,2\' \
        model.num_layers={NUM_LAYERS} \
        model.hidden_size={HIDDEN_SIZE} \
        model.num_attention_heads={NUM_ATTENTION_HEADS} \
        model.activations_checkpoint_method=\'block\' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[tabular_data_text_document]
        
```
If this block does not work in notebook, run it in terminal and change the
Hyperparameters into proper GPU setting, and the path link to the datafolder.(TENSOR_MP_SIZE x PIPLINE_MP_SIZE = NUM_GPUS)
```
NUM_LAYERS = 4
NUM_GPUS = 4
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 8
SEQ_LENGTH = 1024
TENSOR_MP_SIZE = 2
PIPELINE_MP_SIZE = 2
```

Also can resume from checkpoint by using 

==model.resume_from_checkpoint\={PATH_TO_YOUR_CHECKPOINT_FILE}==

```
python megatron_gpt_pretraining.py \
        trainer.devices=4 \
        trainer.accelerator=gpu \
        trainer.log_every_n_steps=100 \
        trainer.val_check_interval=500 \
        trainer.accumulate_grad_batches=1 \
        trainer.max_steps=200000 \
        trainer.precision=16 \
        trainer.gradient_clip_val=1.0 \
        exp_manager.exp_dir=gpt_creditcard_results \
        model.tensor_model_parallel_size=2 \
        model.pipeline_model_parallel_size=2 \
        model.optim.name=fused_adam \
        model.optim.lr=2e-4 \
        model.optim.sched.warmup_steps=2 \
        model.optim.sched.constant_steps=2 \
        model.optim.sched.min_lr=8e-5 \
        model.max_position_embeddings=1024 \
        model.encoder_seq_length=1024 \
        model.data.seq_length=1024 \
        model.tokenizer.type=Tabular \
        model.tokenizer.library=tabular \
        model.tokenizer.vocab_file=Cohort_data/HT_Pros_coder.pickle \
        model.tokenizer.delimiter=\',\' \
        model.data.eod_mask_loss=True \
        model.data.splits_string=\'3800,198,2\' \
        model.num_layers=4 \
        model.hidden_size=1024 \
        model.num_attention_heads=8 \
        model.activations_checkpoint_method=\'block\' \
        model.activations_checkpoint_num_layers=1 \
        model.data.data_prefix=[Cohort_last_tabular_data]
        model.resume_from_checkpoint={PATH_TO_YOUR_CHECKPOINT_FILE}
```
## 6.Convert checkpoint to nemo file
To convert ckpt file, and use different settings for different Number of GPUS you use
For 4 GPUS:
```
python -m torch.distributed.launch --nproc_per_node=4 megatron_ckpt_to_nemo.py \
    --checkpoint_folder=gpt_cohort_results/megatron_gpt/checkpoints \
    --checkpoint_name='megatron_gpt--val_loss=0.09-step=226500-consumed_samples=1812000.0.ckpt' \
    --nemo_file_path=Cohort_tabular.nemo \
    --tensor_model_parallel_size=2 \
    --pipeline_model_parallel_size=2 \
    --gpus_per_node=4 \
    --model_type=gpt
```

## 7.Generate synthetic credit card transactions
Run this block in another terminal to create a server, and remember to change gpt and parallel_sizes for different number of GPUs
```
python megatron_gpt_eval.py \
    gpt_model_file=Cohort_tabular.nemo \
    trainer.devices={GPU_NUM} \
    trainer.num_nodes=1 \
    tensor_model_parallel_size={2 for 4GPU, 1 for 1GPU} \
    pipeline_model_parallel_size={2 for 4GPU, 1 for 1GPU} \
    prompts=[\'\',\'\'] \
    server=True
```
After run this code block, run the 7.Generate synthetic credit card transactions in the notebook

## 8.Evaluation
To generate more sythetic rows and make them into a file, please run the following code block:
```
with open('Cohort_data/HT_Pros_coder.pickle', 'rb') as handle:
        cc: ColumnCodes = pickle.load(handle)

        
        
OUTPUT_FILE_NAME = 'synthetic_csv.csv'
NUM_OF_ROWS = 1000

token_per_rows = sum(cc.sizes) + 1
#token_per_rows = 545
batch_size = 1
port_num = 5555
num_of_rows = 1
headers = {"Content-Type": "application/json"}


def request_data(data):
    
    resp = requests.put('http://localhost:{}/generate'.format(port_num),
                        data=json.dumps(data), headers=headers)
    
    sentences = resp.json()['sentences']
    return sentences


# generate the initial transactions 
data = {
    "sentences": [""] * batch_size,
    "tokens_to_generate": num_of_rows * token_per_rows,
    "temperature": 1.0,
    "add_BOS": True
}
df_out = pd.DataFrame(columns=columns.tolist())
for _ in range(NUM_OF_ROWS):
    sentences = request_data(data)
    for i in range(batch_size):
        if sentences[i][:13] == '<|endoftext|>':
                sentences[i] = sentences[i][13:]
        s = sentences[i][:-2].split(',')
        # print(s)
        df_out.loc[len(df_out.index)] = s
df_out.to_csv(OUTPUT_FILE_NAME)
df_out.head()
```
and use these functions to evaluate the model, which check the coarse grain level evaluation(Duplication of synthetic data)
```
total = pd.concat([df, df_out])
copies = len(total) - len(total.drop_duplicates()) - (len(df) - len(df.drop_duplicates())) - (len(df_out) - len(df_out.drop_duplicates()))
print(copies)
```

```
100*(len(df_out) - len(df_out.drop_duplicates()))/len(df_out)
```

