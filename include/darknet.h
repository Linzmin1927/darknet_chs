#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

//激活函数一共14种
typedef enum{
    LOGISTIC, 
    RELU, 
    RELIE, 
    LINEAR, 
    RAMP, 
    TANH, 
    PLSE, 
    LEAKY, 
    ELU, 
    LOGGY, 
    STAIR, 
    HARDTAN, 
    LHTAN, 
    SELU
} ACTIVATION;
//图片类型4种
typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, 
    MASKED, 
    L1, 
    SEG, 
    SMOOTH,
    WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;
//darknet中网络层的通用结构体
struct layer{
    LAYER_TYPE type;                /* 网络层的类型，枚举类型，取值比如DROPOUT,CONVOLUTIONAL,MAXPOOL
                                     * 分别表示dropout层，卷积层，最大池化层，可参见LAYER_TYPE枚举类
                                     * 型的定义
                                     */
    ACTIVATION activation;          /* 激活函数类型，一共14种 */
    COST_TYPE cost_type;            /* 代价函数类型，一共6种 */
    void (*forward)   (struct layer, struct network);    /* 前向传播函数 */
    void (*backward)  (struct layer, struct network);    /* 反向传播函数 */
    void (*update)    (struct layer, update_args);       /* 权值更新函数 */
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);   
    int batch_normalize;            /* 是否进行BN，如果进行BN，则值为1 */
    int shortcut;                   /* 是否为shortcut层，残差模块中使用*/
    int batch;                      /* 一个batch中含有的图片张数，等于net.batch，
                                     * 详细可以参考network.h中的注释，一般在构建
                                     * 具体网络层时赋值（比如make_maxpool_layer()中）
                                     */
    int forced;                     /**/
    int flipped;                    /**/
    int inputs;                     /* 一张输入图片所含的元素个数（一般在各网络层构
                                     * 建函数中赋值，比如make_connected_layer()），第
                                     * 一层的值等于l.h*l.w*l.c，之后的每一层都是由上一
                                     * 层的输出自动推算得到的（参见parse_network_cfg()，
                                     * 在构建每一层后，会更新params.inputs为上一层的
                                     * l.outputs）
                                     */
    int outputs;                    /* 该层对应一张输入图片的输出元素个数（一般在各网络
                                     * 层构建函数中赋值，比如make_connected_layer()）对于
                                     * 一些网络，可由输入图片的尺寸及相关参数计算出，比如
                                     * 卷积层，可以通过输入尺寸以及跨度、核大小计算出；对于
                                     * 另一些尺寸，则需要通过网络配置文件指定，如未指定，取
                                     * 默认值1，比如全连接层（见parse_connected()函数）
                                     */
	int nweights;                   /* 该卷积层总的权重元素个数（权重元素个数等于输入数据的
									 * 通道数*卷积核个数*卷积核的二维尺寸/分组数，注意因为每
									 * 一个卷积核是同时作用于输入数据的多个通道上的，因此实
									 * 际上卷积核是三维的，包括两个维度的平面尺寸，以及输入
									 * 数据通道数这个维度，每个通道上的卷积核参数都是独立的
									 * 训练参数）
									 */
    int nbiases;                    /* 权重偏置的数量，WX+b中的b的个数，与卷积核个数一致*/
    int extra;                      /**/
    int truths;                     /* 根据region_layer.c判断，这个变量表示一张图片含有的真实
                                     * 值的个数，对于检测模型来说，一个真实的标签含有5个值，
                                     * 包括类型对应的编号以及定位矩形框用到的w,h,x,y四个参数，
                                     * 且在darknet中，固定每张图片最大处理30个矩形框（可查看
                                     * max_boxes参数）,因此，在region_layer.c的make_region_layer()
                                     * 函数中，赋值为30*5
                                     */
    int h,w,c;                      /* 输入 高度，宽度，通道 
                                     * 该层输入图片的高、宽、通道数（一般在各网络层构建函数中
                                     * 赋值，比如make_connected_layer()），第一层网络的h,w,c就是
                                     * 网络初始能够的接收的图片尺寸，而后每一层的h,w,c都与自动匹
                                     * 配上一层相应的输出参数，不再需要配置文件指定（参见
                                     * parse_network_cfg()，在构建每一层后，会更新params.h,params.w,
                                     * params.c及params.inputs为上一层相应的输出参数）,对于全连接层，
                                     * h,w直接置为1,c置为l.inputs（参见make_connected_layer()）
                                     */
    int out_h, out_w, out_c;        /* 输出 高度，宽度，通道 
                                     * 该层输出图片的高、宽、通道数（一般在各网络层构建函数中赋值，
                                     * 比如make_connected_layer()），对于卷积层，可由上面的h,w,c以
                                     * 及卷积核尺寸、跨度计算出；对于全连接层，out_h,out_w的值直接
                                     * 置为1,out_c直接置为l.outputs（参见make_connected_layer()）
                                     */

    
    int n;                          /* 对于卷积层，该参数表示卷积核个数，等于out_c，其值由网络配置文件指定；
                                     * 对于region_layerc层，该参数等于配置文件中的num值
                                     * (该参数通过make_region_layer()函数赋值，而在parser.c中调用的
                                     * make_region_layer()函数)，可以在darknet/cfg文件夹下执行
                                     * 命令：grep num *.cfg便可以搜索出所有设置了num参数的网络，这
                                     * 里面包括yolo.cfg等，其值有设定为3,5,2的，该参数就是Yolo论文中的B，
                                     * 也就是一个cell中预测多少个box。
                                     */
    int max_boxes;                  /* 每张图片最多含有的标签矩形框数（参看：data.c中
                                     * 的load_data_detection()，其输入参数boxes就是指这
                                     * 个参数），就是每张图片中最多打了max_boxes个标签
                                     * 物体，模型预测过程中，可能会预测出很多的物体，
                                     * 但实际上，图片中打上标签的真正存在的物体最多就
                                     * max_boxes个，预测多出来的肯定存在false positive，
                                     * 需要滤出与筛选，可参看region_layer.c中forward_region_layer()
                                     * 函数的第二个for循环中的注释
                                     */
    int groups;                     /* 分组卷积分组数，softmax中分组计算的分组数 
                                     * 在softmax中含义是将一张图片的数据分成几组，
                                     * 具体的值由网络配置文件指定，如未指定默认为1
                                     * （见parse_softmax()），很多网络都将该值设置为1，
                                     * 相当于没用到该值，我想这可能跟分类与分割粒度
                                     * 有关（如果粒度细些，估计会大于1,未验证！！！）
                                     */
    int size;                       /* 核尺寸（比如卷积核，池化核等）*/
    int side;                       /**/
    int stride;                     /**/
    int reverse;                    /**/
    int flatten;                    /**/
    int spatial;                    /**/
    int pad;                        /* 四周补0长度，卷积层，最大池化层中使用*/
    int sqrt;                       /* 输出尺寸是否平方，为1，这输出w 变为w*w 输入h变为 h*h*/
    int flip;                       /**/
    int index;                      /**/
    int binary;                     /* 是否对权重进行二值化 */
    int xnor;                       /* 是否对权重以及输入进行二值化*/
    int steps;                      /**/
    int hidden;                     /**/
    int truth;                      /**/
    float smooth;                   /**/
    float dot;                      /**/
    float angle;                    /**/
    float jitter;                   /**/
    float saturation;               /**/
    float exposure;                 /**/
    float shift;                    /**/
    float ratio;                    /**/
    float learning_rate_scale;      /**/
    float clip;                     /**/
    int noloss;                     /**/
    int softmax;                    /**/
    int classes;                    /* 物体类别种数，一个训练好的网络，只能检测指定所有
                                     * 物体类别中的物体，比如yolo9000.cfg，设置该值为9418，
                                     * 也就是该网络训练好了之后可以检测9418种物体。该参数
                                     * 由网络配置文件指定。目前在作者给的例子中，有设置该
                                     * 值的配置文件大都是检测模型，纯识别的网络模型没有设
                                     * 置该值，我想是因为检测模型输出的一般会为各个类别的
                                     * 概率，所以需要知道这个种类数目，而识别的话，不需要
                                     * 知道某个物体属于这些所有类的具体概率，因此可以不知道。
                                     */
    int coords;                     /* 这个参数一般用在检测模型中，且不是所有层都有这个参数，
                                     * ，比如region_layer层，该参数的含义  是定位一个物体所需
                                     * 的参数个数，一般为4个，包括物体所在矩形框中心坐标x,y
                                     * 两个参数以及矩形框长宽w,h两个参数，可以在darknet/cfg
                                     * 文件夹下，执行grep coords *.cfg，会搜索出所有使用该参
                                     * 数的模型，并可看到该值都设置位4
                                     */
    int background;                 /**/
    int rescore;                    /**/
    int objectness;                 /**/
    int joint;                      /**/
    int noadjust;                   /**/
    int log;                        /**/
    int reorg;                      /**/
    int tanh;                       /**/
    int *mask;                      /**/
    int total;                      /**/

    float alpha;                    /**/
    float beta;                     /**/
    float kappa;                    /**/

    float coord_scale;              /**/
    float object_scale;             /**/
    float noobject_scale;           /**/
    float mask_scale;               /**/
    float class_scale;              /**/
    int bias_match;                 /**/
    int random;                     /**/
    float ignore_thresh;            /**/
    float truth_thresh;             /**/
    float thresh;                   /**/
    float focus;                    /**/
    int classfix;                   /**/
    int absolute;                   /**/

    int onlyforward;                /**/
    int stopbackward;               /* 标志参数?不确定，用来强制停止反向传播过程
                                     *（值为1则停止反向传播），参看network.c
                                     * 中的backward_network()函数
                                     */
    int dontload;                   /**/
    int dontsave;                   /**/
    int dontloadscales;             /**/
    int numload;                    /**/

    float temperature;              /* 温度参数，softmax层特有参数，在parse_softmax()
                                     * 函数中赋值，由网络配置文件指定，如果未指定，则
                                     * 使用默认值1（见parse_softmax()）
                                     */
    float probability;              /* dropout概率，即舍弃概率，相应的1-probability为保
                                     * 留概率（具体的使用可参见forward_dropout_layer()），
                                     * 在make_dropout_layer()中赋值，其值由网络配置文件
                                     * 指定，如果网络配置文件未指定，则取默认值0.5
                                     * （见parse_dropout()）
                                     */
    float scale;                    /* 在dropout层中，该变量是一个比例因子，取值为保留概率
                                     * 的倒数（darknet实现用的是inverted dropout），用于缩放
                                     * 输入元素的值
                                     * 在crop_layer 中，表示输出wh与输入wh的缩放比例
									 * 在卷积层是权重初始化的系数为sqrt(2./(size*size*c/l.groups))，msra初始化
                                     */

    char  * cweights;               /* 开启二值化(binary)时分配空间，大小为nweights，源码中没用用到该空间！*/
    int   * indexes;                /**/
    int   * input_layers;           /**/
    int   * input_sizes;            /**/
    int   * map;                    /* 这个参数用的不多，仅在region_layer.c中使用，该参数
                                     * 的作用是用于不同数据集间类别编号的转换，更为具体的，
                                     * 是coco数据集中80类物体编号与联合数据集中9000+物体类
                                     * 别编号之间的转换，可以对比查看data/coco.names与
                                     * data/9k.names以及data/coco9k.map三个文件（旧版的darknet
                                     * 可能没有，新版的darknet才有coco9k.map这个文件），可以发
                                     * 现，coco.names中每一个物体类别都可以在9k.names中找到,且
                                     * coco.names中每个物体类别名称在9k.names中所在的行数就是
                                     * coco9k.map中的编号值（减了1,因为在程序数组中编号从0开始），
                                     * 也就是这个map将coco数据集中的类别编号映射到联和数据集9k
                                     * 中的类别编号（这个9k数据集是一个联和多个数据集的大数集，
                                     * 其名称分类被层级划分）（注意两个文件中物体的类别名称大部
                                     * 分都相同，有小部分存在小差异，虽然有差异，但只是两个数据
                                     * 集中使用的名称有所差异而已，对应的物体是一样的，比如在
                                     * coco.names中摩托车的名称为motorbike，在联合数据集9k.names，
                                     * 其名称为motorcycle）.                   
                                     */
    
    int   * counts;                 /**/
    float ** sums;                  /**/
    float * rand;                   /**/
    float * cost;                   /* 目标函数值，该参数不是所有层都有的，一般在网络最后一层拥有，
                                     * 用于计算最后的cost，比如识别模型中的cost_layer层，检测模型中
                                     * 的region_layer层
                                     */
    float * state;                  /**/
    float * prev_state;             /**/
    float * forgot_state;           /**/
    float * forgot_delta;           /**/
    float * state_delta;            /**/
    float * combine_cpu;            /**/
    float * combine_delta_cpu;      /**/

    float * concat;                 /**/
    float * concat_delta;           /**/

    float * binary_weights;         /* 启用二值化（binary）时的，二值化权重空间*/

    float * biases;                 /* 指向偏置存储空间，bias就是Wx+b中的b（weights就是W），有多少
									 * 个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为
									 * c*size*size）
									 */
    float * bias_updates;           /* 指向bias更新暂存空间，大小与biases一致 */

    float * scales;                 /* 开启二值化或批标准化时，指向存放缩放系数的内存空间
									 * batch_normalize中初始值全为1	，大小为卷积核数*sizeof(float)
									 */
    float * scale_updates;          /* 开启批标准化时，指向存放缩放系数更新的内存空间，大小与scales一致*/

    float * weights;                /* 指向权重存储空间，空间大小为该卷积层总的权重元素（卷积核元素）个数
									 * =输入图像通道数/分组数*卷积核个数/分组数*卷积核尺寸*分组数
									 */
    float * weight_updates;         /* 指向权重更新暂存空间，大小与weights一致 */

    float * delta;                  /* 指向当前层给个变量的梯度存储空间，大小与output一致 */
    float * output;                 /* 指向该层输出存储空间， l.output为该层所有的输出
									 *（包括mini-batch所有输入图片的输出）
									 */
    float * loss;                   /* 指向loss值暂存空间，logistic_layer与*softmax_layer中应用，大小为输入元素数量*batch*/
    float * squared;                /* normalization_layer中暂存空间，其他地方没看到引用*/
    float * norms;                  /* normalization_layer中引用暂存空间，其他地方没看到引用*/

    float * spatial_mean;           /* 没用到 */
    float * mean;                   /* batch_normalize 中均值*/
    float * variance;               /* batch_normalize 中方差*/

    float * mean_delta;             /* batch_normalize计算中参数*/
    float * variance_delta;         /* batch_normalize计算中参数*/

    float * rolling_mean;           /* batch_normalize计算中参数*/
    float * rolling_variance;       /* batch_normalize计算中参数*/

    float * x;                  	/* batch_normalize计算中参数*/
    float * x_norm;                 /* batch_normalize计算中参数*/

    float * m;                  	/* adam中的参数 动量*/
    float * v;                  	/* adam中的参数*/
    
    float * bias_m;                 /* adam中的参数*/
    float * bias_v;                 /* adam中的参数*/
    float * scale_m;                /* adam中的参数*/
    float * scale_v;                /* adam中的参数*/

	/* 此部分为RNN/LSTM/GRU 模型参数*/
    float *z_cpu;                  	/**/
    float *r_cpu;                  	/**/
    float *h_cpu;                  	/**/
    float * prev_state_cpu;         /**/

    float *temp_cpu;                /**/
    float *temp2_cpu;               /**/
    float *temp3_cpu;               /**/

    float *dh_cpu;                  /**/
    float *hh_cpu;                  /**/
    float *prev_cell_cpu;           /**/
    float *cell_cpu;                /**/
    float *f_cpu;                   /**/
    float *i_cpu;                   /**/
    float *g_cpu;                   /**/
    float *o_cpu;                   /**/
    float *c_cpu;                   /**/
    float *dc_cpu;                  /**/

    float * binary_input;           /* 开启xnor后，指向输入二值化存储空间*/

	/* 此部分为RNN/LSTM/GRU 模型参数*/
    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;               /**/
    struct layer *uz;               /**/
    struct layer *wr;               /**/
    struct layer *ur;               /**/
    struct layer *wh;               /**/
    struct layer *uh;               /**/
    struct layer *uo;               /**/
    struct layer *wo;               /**/
    struct layer *uf;               /**/
    struct layer *wf;               /**/
    struct layer *ui;               /**/
    struct layer *wi;               /**/
    struct layer *ug;               /**/
    struct layer *wg;               /**/

    tree *softmax_tree;             /* softmax参数 */

    size_t workspace_size;          /* Conv/Deconv中返回输出内存空间的大小
									 * local_layer 中返回输出元素个数
								     */

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;                          /* 网络总层数（make_network()时赋值）*/
    int batch;                      /* 一个batch含有的图片张数parse_net_options()中赋值
                                     * 此处的batch*subdivision才等于网络配置文件中指定的batch值
                                     */
    size_t *seen;                   /* 目前已经读入的图片张数（网络已经处理的图片张数）
                                     * （在make_network()中动态分配内存）
                                     */
    int *t;                         /**/
    float epoch;                    /**/
    int subdivisions;               /* 执行计算的线程数？parse_net_options()中赋值*/
    layer *layers;                  /* 存储网络所有的层，在make_network()中动态分配内存*/
    float *output;                  /**/
    learning_rate_policy policy;    /* 学习率衰减策略*/

    float learning_rate;            /* 学习率，parse_net_options()中赋值*/
    float momentum;                 /* 梯度下降动量*/
    float decay;                    /* lr衰减系数*/
    float gamma;                    /**/
    float scale;                    /**/
    float power;                    /**/
    int time_steps;                 /**/
    int step;                       /**/
    int max_batches;                /* 最大训练多少批次/所有数据最多可以有多少批次 */
    float *scales;                  /**/
    int   *steps;                   /**/
    int num_steps;                  /**/
    int burn_in;                    /**/

    int adam;                       /**/
    float B1;                       /**/
    float B2;                       /**/
    float eps;                      /**/

    int inputs;                     /*输入元素数量 w*h*c */
    int outputs;                    /*一张输入图片对应的输出元素个数，对于一些网络，可由输入图片的尺寸
                                     *及相关参数计算出，比如卷积层，可以通过输入尺寸以及跨度、核大小计
                                     *算出；对于另一些尺寸，则需要通过网络配置文件指定，如未指定，取默
                                     *认值1，比如全连接层
                                     */
    int truths;                     /**/
    int notruth;                    /**/
    int h, w, c;                    /**/
    int max_crop;                   /**/
    int min_crop;                   /**/
    float max_ratio;                /**/
    float min_ratio;                /**/
    int center;                     /**/
    float angle;                    /**/
    float aspect;                   /**/
    float exposure;                 /**/
    float saturation;               /**/
    float hue;                      /**/
    int random;                     /**/

    int gpu_index;                  /*gpu设备号，为-1表示不用GPU*/
    tree *hierarchy;                /**/

    float *input;                   /*中间变量，用来暂存某层网络的输入（包含一个batch的输入，
                                     *比如某层网络完成前向，将其输出赋给该变量，作为下一层的
                                     *输入，可以参看network.c中的forward_network()与
                                     *backward_network()两个函数），当然，也是网络接受最原始
                                     *输入数据（即第一层网络接收的输入）的变量（比如在图像检
                                     *测训练中，最早在train_detector()->train_network()->get_next_batch()
                                     *函数中赋值）
                                     */
    float *truth;                   /*中间变量，与上面的input对应，用来暂存input数据对应的标签
                                     *数据（真实数据）存储一个batch中所有图片所有box的[x y w h classid]
                                     */
    float *delta;                   /*中间变量，用来暂存某层网络的梯度（反向传播处理当前层时，
                                     *用来存储上一层的梯度图，因为当前层会计算部分上一层的梯
                                     *度图，可以参看network.c中的backward_network()函数），net.delta
                                     *并没有在创建网络之初就为其动态分配了内存，而是等到反向传播时，
                                     *直接将其等于某一层的l.delta（l.delta是在创建每一层网络之初就
                                     *动态为其分配了内存），这才为net.delta分配了内存，如果
                                     *没有令net.delta=l.delta，则net.delta是未定义的（没有动态分配内存的）
                                     */
    float *workspace;               /* 整个网络的工作空间，其元素个数为所有层中最大
                                     *的l.workspace_size = l.out_h*l.out_w*l.size*l.size*l.c
                                     *（在make_convolutional_layer()计算得到workspace_size的
                                     *大小，在parse_network_cfg()中动态分配内存，此值对应未
                                     *使用gpu时的情况），该变量貌似不轻易被释放内存，目前只
                                     *发现在network.c的resize_network()函数对其进行了释放。
                                     *net.workspace充当一个临时工作空间的作用，存储临时所需
                                     *要的计算参数，比如每层单张图片重排后的结果（这些参数
                                     *马上就会参与卷积运算），一旦用完，就会被马上更新（因
                                     *此该变量的值的更新频率比较大）
                                     */
                                                                     
    int train;                      /*标志参数，网络是否处于训练阶段，如果是，则值为1（这个
                                     *参数一般用于训练与测试有不同操作的情况，比如dropout层，
                                     *对于训练，才需要进行forward_dropout_layer()函数，对于测
                                     试，不需要进入到该函数）
                                     */
    int index;                      /*标志参数，当前网络的活跃层（活跃包括前向和反向，可参
                                     *考network.c中forward_network()与backward_network()函数
                                     */
    float *cost;                    /**/
    float clip;                     /**/

#ifdef GPU
    float *input_gpu;               /**/
    float *truth_gpu;               /**/
    float *delta_gpu;               /**/
    float *output_gpu;              /**/
#endif

} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

/* 导入数据的时候的数据结构体 */
typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;     /* shallow是指深层释放X,y中vals的内存还是浅层释放X,y中vals
                      * （注意是X,y的vals元素，不是X,y本身，X,y本身是万不能用
                      * free释放的，因为其连指针都不是）的内存X,y的vals是一个
                      * 二维数组，如果浅层释放，即直接释放free(vals)，这种释放
                      * 并不会真正释放二维数组的内容，因为二维数组实际是指针的
                      * 指针，这种直接释放只是释放了用于存放第二层指针变量的内
                      * 存，也就导致二维数组的每一行将不再能通过二维数组名来访
                      * 问了（因此这种释放，需要先将数据转移，使得有其他指针能
                      * 够指向这些数据块，不能就会造成内存溢出了）；
                      * 深层释放，是循环逐行释放为每一行动态分配的内存，然后再
                      * 释放vals，释放完后，整个二维数组包括第一层存储的第二维
                      * 指针变量以及实际数据将都不再存在，所有数据都被清空。详
                      * 细可查看free_data()以及free_matrix()函数。
                      */
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, 
    DETECTION_DATA, 
    CAPTCHA_DATA, 
    REGION_DATA, 
    IMAGE_DATA, 
    COMPARE_DATA, 
    WRITING_DATA, 
    SWAG_DATA, 
    TAG_DATA, 
    OLD_CLASSIFICATION_DATA, 
    STUDY_DATA, 
    DET_DATA, 
    SUPER_DATA, 
    LETTERBOX_DATA, 
    REGRESSION_DATA, 
    SEGMENTATION_DATA, 
    INSTANCE_DATA, 
    ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

/*
** 图片检测标签数据：图片检测包括识别与定位，定位通过一个矩形框来实现，
** 因此，对于图片检测，标签数据依次包括：物体类别id，矩形框中心点x,y坐标，
** 矩形框宽高，以及矩形框四个角点的最小最大x,y坐标
*/
typedef struct{
    int id;                             // 矩形框类别（即矩形框框起来的物体的标签/类别id）
    float x,y,w,h;                      // 矩形中心点的x,y坐标，以及矩形宽高w,h（值不是真实的像素坐标，而是相对输入图片的宽高比例）
    float left, right, top, bottom;     // 矩形四个角点的最大最小x,y坐标（值不是真实的像素坐标，而是相对输入图片的宽高比例）
} box_label;



network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
float train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix csv_to_matrix(char *filename);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, float t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image float_to_image(int w, int h, int c, float *data);
void ghost_image(image source, image dest, int dx, int dy);
float network_accuracy(network *net, data d);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
double what_time_is_it_now();
image rotate_image(image m, float rad);
void visualize_network(network *net);
float box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box float_to_box(float *f, int stride);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
float *network_predict(network *net, float *input);

int network_width(network *net);
int network_height(network *net);
float *network_predict_image(network *net, image im);
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
float train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif
#endif
