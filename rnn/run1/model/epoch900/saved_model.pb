шь
§%Ж%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	АР
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	Р
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
1
Square
x"T
y"T"
Ttype:

2	
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.13.12
b'unknown'Ёґ
l
XPlaceholder*
dtype0*+
_output_shapes
:€€€€€€€€€
* 
shape:€€€€€€€€€

d
YPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
С
input/unstackUnpackX*
T0*	
num
*

axis*‘
_output_shapesЅ
Њ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
Z
rnn/rnn/ShapeShapeinput/unstack*
T0*
out_type0*
_output_shapes
:
e
rnn/rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
g
rnn/rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
g
rnn/rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
°
rnn/rnn/strided_sliceStridedSlicernn/rnn/Shapernn/rnn/strided_slice/stackrnn/rnn/strided_slice/stack_1rnn/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
o
-rnn/rnn/BasicLSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ѓ
)rnn/rnn/BasicLSTMCellZeroState/ExpandDims
ExpandDimsrnn/rnn/strided_slice-rnn/rnn/BasicLSTMCellZeroState/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:
o
$rnn/rnn/BasicLSTMCellZeroState/ConstConst*
valueB:А*
dtype0*
_output_shapes
:
l
*rnn/rnn/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
и
%rnn/rnn/BasicLSTMCellZeroState/concatConcatV2)rnn/rnn/BasicLSTMCellZeroState/ExpandDims$rnn/rnn/BasicLSTMCellZeroState/Const*rnn/rnn/BasicLSTMCellZeroState/concat/axis*

Tidx0*
T0*
N*
_output_shapes
:
o
*rnn/rnn/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ƒ
$rnn/rnn/BasicLSTMCellZeroState/zerosFill%rnn/rnn/BasicLSTMCellZeroState/concat*rnn/rnn/BasicLSTMCellZeroState/zeros/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
q
/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
≤
+rnn/rnn/BasicLSTMCellZeroState/ExpandDims_1
ExpandDimsrnn/rnn/strided_slice/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes
:
q
&rnn/rnn/BasicLSTMCellZeroState/Const_1Const*
valueB:А*
dtype0*
_output_shapes
:
q
/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
≤
+rnn/rnn/BasicLSTMCellZeroState/ExpandDims_2
ExpandDimsrnn/rnn/strided_slice/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_2/dim*

Tdim0*
T0*
_output_shapes
:
q
&rnn/rnn/BasicLSTMCellZeroState/Const_2Const*
valueB:А*
dtype0*
_output_shapes
:
n
,rnn/rnn/BasicLSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
р
'rnn/rnn/BasicLSTMCellZeroState/concat_1ConcatV2+rnn/rnn/BasicLSTMCellZeroState/ExpandDims_2&rnn/rnn/BasicLSTMCellZeroState/Const_2,rnn/rnn/BasicLSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
q
,rnn/rnn/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
 
&rnn/rnn/BasicLSTMCellZeroState/zeros_1Fill'rnn/rnn/BasicLSTMCellZeroState/concat_1,rnn/rnn/BasicLSTMCellZeroState/zeros_1/Const*
T0*

index_type0*(
_output_shapes
:€€€€€€€€€А
q
/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_3/dimConst*
value	B : *
dtype0*
_output_shapes
: 
≤
+rnn/rnn/BasicLSTMCellZeroState/ExpandDims_3
ExpandDimsrnn/rnn/strided_slice/rnn/rnn/BasicLSTMCellZeroState/ExpandDims_3/dim*
T0*
_output_shapes
:*

Tdim0
q
&rnn/rnn/BasicLSTMCellZeroState/Const_3Const*
valueB:А*
dtype0*
_output_shapes
:
ї
;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
valueB"     *
dtype0*
_output_shapes
:
≠
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
valueB
 *ЙМљ*
dtype0*
_output_shapes
: 
≠
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
valueB
 *ЙМ=*
dtype0*
_output_shapes
: 
С
Crnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform;rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
ВА*

seed *
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
Ж
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
_output_shapes
: 
Ъ
9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulCrnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
ВА*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
М
5rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul9rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*-
_class#
!loc:@rnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА*
T0
Ѕ
rnn/basic_lstm_cell/kernel
VariableV2*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
	container *
shape:
ВА*
dtype0* 
_output_shapes
:
ВА
Б
!rnn/basic_lstm_cell/kernel/AssignAssignrnn/basic_lstm_cell/kernel5rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
r
rnn/basic_lstm_cell/kernel/readIdentityrnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА*
T0
≤
:rnn/basic_lstm_cell/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*+
_class!
loc:@rnn/basic_lstm_cell/bias*
valueB:А*
dtype0
Ґ
0rnn/basic_lstm_cell/bias/Initializer/zeros/ConstConst*+
_class!
loc:@rnn/basic_lstm_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
Е
*rnn/basic_lstm_cell/bias/Initializer/zerosFill:rnn/basic_lstm_cell/bias/Initializer/zeros/shape_as_tensor0rnn/basic_lstm_cell/bias/Initializer/zeros/Const*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*

index_type0*
_output_shapes	
:А
≥
rnn/basic_lstm_cell/bias
VariableV2*
dtype0*
_output_shapes	
:А*
shared_name *+
_class!
loc:@rnn/basic_lstm_cell/bias*
	container *
shape:А
л
rnn/basic_lstm_cell/bias/AssignAssignrnn/basic_lstm_cell/bias*rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
i
rnn/basic_lstm_cell/bias/readIdentityrnn/basic_lstm_cell/bias*
T0*
_output_shapes	
:А
_
rnn/rnn/basic_lstm_cell/ConstConst*
_output_shapes
: *
value	B :*
dtype0
e
#rnn/rnn/basic_lstm_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ќ
rnn/rnn/basic_lstm_cell/concatConcatV2input/unstack&rnn/rnn/BasicLSTMCellZeroState/zeros_1#rnn/rnn/basic_lstm_cell/concat/axis*
T0*
N*(
_output_shapes
:€€€€€€€€€В*

Tidx0
¬
rnn/rnn/basic_lstm_cell/MatMulMatMulrnn/rnn/basic_lstm_cell/concatrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( 
≥
rnn/rnn/basic_lstm_cell/BiasAddBiasAddrnn/rnn/basic_lstm_cell/MatMulrnn/basic_lstm_cell/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
a
rnn/rnn/basic_lstm_cell/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
ж
rnn/rnn/basic_lstm_cell/splitSplitrnn/rnn/basic_lstm_cell/Constrnn/rnn/basic_lstm_cell/BiasAdd*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
d
rnn/rnn/basic_lstm_cell/Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ч
rnn/rnn/basic_lstm_cell/AddAddrnn/rnn/basic_lstm_cell/split:2rnn/rnn/basic_lstm_cell/Const_2*
T0*(
_output_shapes
:€€€€€€€€€А
z
rnn/rnn/basic_lstm_cell/SigmoidSigmoidrnn/rnn/basic_lstm_cell/Add*(
_output_shapes
:€€€€€€€€€А*
T0
Ь
rnn/rnn/basic_lstm_cell/MulMul$rnn/rnn/BasicLSTMCellZeroState/zerosrnn/rnn/basic_lstm_cell/Sigmoid*(
_output_shapes
:€€€€€€€€€А*
T0
~
!rnn/rnn/basic_lstm_cell/Sigmoid_1Sigmoidrnn/rnn/basic_lstm_cell/split*
T0*(
_output_shapes
:€€€€€€€€€А
x
rnn/rnn/basic_lstm_cell/TanhTanhrnn/rnn/basic_lstm_cell/split:1*
T0*(
_output_shapes
:€€€€€€€€€А
Ш
rnn/rnn/basic_lstm_cell/Mul_1Mul!rnn/rnn/basic_lstm_cell/Sigmoid_1rnn/rnn/basic_lstm_cell/Tanh*(
_output_shapes
:€€€€€€€€€А*
T0
У
rnn/rnn/basic_lstm_cell/Add_1Addrnn/rnn/basic_lstm_cell/Mulrnn/rnn/basic_lstm_cell/Mul_1*(
_output_shapes
:€€€€€€€€€А*
T0
x
rnn/rnn/basic_lstm_cell/Tanh_1Tanhrnn/rnn/basic_lstm_cell/Add_1*
T0*(
_output_shapes
:€€€€€€€€€А
А
!rnn/rnn/basic_lstm_cell/Sigmoid_2Sigmoidrnn/rnn/basic_lstm_cell/split:3*(
_output_shapes
:€€€€€€€€€А*
T0
Ъ
rnn/rnn/basic_lstm_cell/Mul_2Mulrnn/rnn/basic_lstm_cell/Tanh_1!rnn/rnn/basic_lstm_cell/Sigmoid_2*
T0*(
_output_shapes
:€€€€€€€€€А
a
rnn/rnn/basic_lstm_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ћ
 rnn/rnn/basic_lstm_cell/concat_1ConcatV2input/unstack:1rnn/rnn/basic_lstm_cell/Mul_2%rnn/rnn/basic_lstm_cell/concat_1/axis*(
_output_shapes
:€€€€€€€€€В*

Tidx0*
T0*
N
∆
 rnn/rnn/basic_lstm_cell/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_1rnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( *
T0
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_1BiasAdd rnn/rnn/basic_lstm_cell/MatMul_1rnn/basic_lstm_cell/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
a
rnn/rnn/basic_lstm_cell/Const_4Const*
_output_shapes
: *
value	B :*
dtype0
м
rnn/rnn/basic_lstm_cell/split_1Splitrnn/rnn/basic_lstm_cell/Const_3!rnn/rnn/basic_lstm_cell/BiasAdd_1*
T0*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А
d
rnn/rnn/basic_lstm_cell/Const_5Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ы
rnn/rnn/basic_lstm_cell/Add_2Add!rnn/rnn/basic_lstm_cell/split_1:2rnn/rnn/basic_lstm_cell/Const_5*(
_output_shapes
:€€€€€€€€€А*
T0
~
!rnn/rnn/basic_lstm_cell/Sigmoid_3Sigmoidrnn/rnn/basic_lstm_cell/Add_2*
T0*(
_output_shapes
:€€€€€€€€€А
Щ
rnn/rnn/basic_lstm_cell/Mul_3Mulrnn/rnn/basic_lstm_cell/Add_1!rnn/rnn/basic_lstm_cell/Sigmoid_3*
T0*(
_output_shapes
:€€€€€€€€€А
А
!rnn/rnn/basic_lstm_cell/Sigmoid_4Sigmoidrnn/rnn/basic_lstm_cell/split_1*
T0*(
_output_shapes
:€€€€€€€€€А
|
rnn/rnn/basic_lstm_cell/Tanh_2Tanh!rnn/rnn/basic_lstm_cell/split_1:1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
rnn/rnn/basic_lstm_cell/Mul_4Mul!rnn/rnn/basic_lstm_cell/Sigmoid_4rnn/rnn/basic_lstm_cell/Tanh_2*
T0*(
_output_shapes
:€€€€€€€€€А
Х
rnn/rnn/basic_lstm_cell/Add_3Addrnn/rnn/basic_lstm_cell/Mul_3rnn/rnn/basic_lstm_cell/Mul_4*
T0*(
_output_shapes
:€€€€€€€€€А
x
rnn/rnn/basic_lstm_cell/Tanh_3Tanhrnn/rnn/basic_lstm_cell/Add_3*(
_output_shapes
:€€€€€€€€€А*
T0
В
!rnn/rnn/basic_lstm_cell/Sigmoid_5Sigmoid!rnn/rnn/basic_lstm_cell/split_1:3*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
rnn/rnn/basic_lstm_cell/Mul_5Mulrnn/rnn/basic_lstm_cell/Tanh_3!rnn/rnn/basic_lstm_cell/Sigmoid_5*
T0*(
_output_shapes
:€€€€€€€€€А
a
rnn/rnn/basic_lstm_cell/Const_6Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ћ
 rnn/rnn/basic_lstm_cell/concat_2ConcatV2input/unstack:2rnn/rnn/basic_lstm_cell/Mul_5%rnn/rnn/basic_lstm_cell/concat_2/axis*
T0*
N*(
_output_shapes
:€€€€€€€€€В*

Tidx0
∆
 rnn/rnn/basic_lstm_cell/MatMul_2MatMul rnn/rnn/basic_lstm_cell/concat_2rnn/basic_lstm_cell/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_2BiasAdd rnn/rnn/basic_lstm_cell/MatMul_2rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
a
rnn/rnn/basic_lstm_cell/Const_7Const*
_output_shapes
: *
value	B :*
dtype0
м
rnn/rnn/basic_lstm_cell/split_2Splitrnn/rnn/basic_lstm_cell/Const_6!rnn/rnn/basic_lstm_cell/BiasAdd_2*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
d
rnn/rnn/basic_lstm_cell/Const_8Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ы
rnn/rnn/basic_lstm_cell/Add_4Add!rnn/rnn/basic_lstm_cell/split_2:2rnn/rnn/basic_lstm_cell/Const_8*(
_output_shapes
:€€€€€€€€€А*
T0
~
!rnn/rnn/basic_lstm_cell/Sigmoid_6Sigmoidrnn/rnn/basic_lstm_cell/Add_4*(
_output_shapes
:€€€€€€€€€А*
T0
Щ
rnn/rnn/basic_lstm_cell/Mul_6Mulrnn/rnn/basic_lstm_cell/Add_3!rnn/rnn/basic_lstm_cell/Sigmoid_6*
T0*(
_output_shapes
:€€€€€€€€€А
А
!rnn/rnn/basic_lstm_cell/Sigmoid_7Sigmoidrnn/rnn/basic_lstm_cell/split_2*
T0*(
_output_shapes
:€€€€€€€€€А
|
rnn/rnn/basic_lstm_cell/Tanh_4Tanh!rnn/rnn/basic_lstm_cell/split_2:1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
rnn/rnn/basic_lstm_cell/Mul_7Mul!rnn/rnn/basic_lstm_cell/Sigmoid_7rnn/rnn/basic_lstm_cell/Tanh_4*(
_output_shapes
:€€€€€€€€€А*
T0
Х
rnn/rnn/basic_lstm_cell/Add_5Addrnn/rnn/basic_lstm_cell/Mul_6rnn/rnn/basic_lstm_cell/Mul_7*
T0*(
_output_shapes
:€€€€€€€€€А
x
rnn/rnn/basic_lstm_cell/Tanh_5Tanhrnn/rnn/basic_lstm_cell/Add_5*
T0*(
_output_shapes
:€€€€€€€€€А
В
!rnn/rnn/basic_lstm_cell/Sigmoid_8Sigmoid!rnn/rnn/basic_lstm_cell/split_2:3*(
_output_shapes
:€€€€€€€€€А*
T0
Ъ
rnn/rnn/basic_lstm_cell/Mul_8Mulrnn/rnn/basic_lstm_cell/Tanh_5!rnn/rnn/basic_lstm_cell/Sigmoid_8*(
_output_shapes
:€€€€€€€€€А*
T0
a
rnn/rnn/basic_lstm_cell/Const_9Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
Ћ
 rnn/rnn/basic_lstm_cell/concat_3ConcatV2input/unstack:3rnn/rnn/basic_lstm_cell/Mul_8%rnn/rnn/basic_lstm_cell/concat_3/axis*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€В
∆
 rnn/rnn/basic_lstm_cell/MatMul_3MatMul rnn/rnn/basic_lstm_cell/concat_3rnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( *
T0
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_3BiasAdd rnn/rnn/basic_lstm_cell/MatMul_3rnn/basic_lstm_cell/bias/read*(
_output_shapes
:€€€€€€€€€А*
T0*
data_formatNHWC
b
 rnn/rnn/basic_lstm_cell/Const_10Const*
_output_shapes
: *
value	B :*
dtype0
м
rnn/rnn/basic_lstm_cell/split_3Splitrnn/rnn/basic_lstm_cell/Const_9!rnn/rnn/basic_lstm_cell/BiasAdd_3*
T0*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А
e
 rnn/rnn/basic_lstm_cell/Const_11Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ь
rnn/rnn/basic_lstm_cell/Add_6Add!rnn/rnn/basic_lstm_cell/split_3:2 rnn/rnn/basic_lstm_cell/Const_11*(
_output_shapes
:€€€€€€€€€А*
T0
~
!rnn/rnn/basic_lstm_cell/Sigmoid_9Sigmoidrnn/rnn/basic_lstm_cell/Add_6*
T0*(
_output_shapes
:€€€€€€€€€А
Щ
rnn/rnn/basic_lstm_cell/Mul_9Mulrnn/rnn/basic_lstm_cell/Add_5!rnn/rnn/basic_lstm_cell/Sigmoid_9*
T0*(
_output_shapes
:€€€€€€€€€А
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_10Sigmoidrnn/rnn/basic_lstm_cell/split_3*(
_output_shapes
:€€€€€€€€€А*
T0
|
rnn/rnn/basic_lstm_cell/Tanh_6Tanh!rnn/rnn/basic_lstm_cell/split_3:1*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_10Mul"rnn/rnn/basic_lstm_cell/Sigmoid_10rnn/rnn/basic_lstm_cell/Tanh_6*(
_output_shapes
:€€€€€€€€€А*
T0
Ц
rnn/rnn/basic_lstm_cell/Add_7Addrnn/rnn/basic_lstm_cell/Mul_9rnn/rnn/basic_lstm_cell/Mul_10*(
_output_shapes
:€€€€€€€€€А*
T0
x
rnn/rnn/basic_lstm_cell/Tanh_7Tanhrnn/rnn/basic_lstm_cell/Add_7*
T0*(
_output_shapes
:€€€€€€€€€А
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_11Sigmoid!rnn/rnn/basic_lstm_cell/split_3:3*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_11Mulrnn/rnn/basic_lstm_cell/Tanh_7"rnn/rnn/basic_lstm_cell/Sigmoid_11*(
_output_shapes
:€€€€€€€€€А*
T0
b
 rnn/rnn/basic_lstm_cell/Const_12Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_4/axisConst*
_output_shapes
: *
value	B :*
dtype0
ћ
 rnn/rnn/basic_lstm_cell/concat_4ConcatV2input/unstack:4rnn/rnn/basic_lstm_cell/Mul_11%rnn/rnn/basic_lstm_cell/concat_4/axis*
N*(
_output_shapes
:€€€€€€€€€В*

Tidx0*
T0
∆
 rnn/rnn/basic_lstm_cell/MatMul_4MatMul rnn/rnn/basic_lstm_cell/concat_4rnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( *
T0
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_4BiasAdd rnn/rnn/basic_lstm_cell/MatMul_4rnn/basic_lstm_cell/bias/read*(
_output_shapes
:€€€€€€€€€А*
T0*
data_formatNHWC
b
 rnn/rnn/basic_lstm_cell/Const_13Const*
_output_shapes
: *
value	B :*
dtype0
н
rnn/rnn/basic_lstm_cell/split_4Split rnn/rnn/basic_lstm_cell/Const_12!rnn/rnn/basic_lstm_cell/BiasAdd_4*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
e
 rnn/rnn/basic_lstm_cell/Const_14Const*
_output_shapes
: *
valueB
 *  А?*
dtype0
Ь
rnn/rnn/basic_lstm_cell/Add_8Add!rnn/rnn/basic_lstm_cell/split_4:2 rnn/rnn/basic_lstm_cell/Const_14*
T0*(
_output_shapes
:€€€€€€€€€А

"rnn/rnn/basic_lstm_cell/Sigmoid_12Sigmoidrnn/rnn/basic_lstm_cell/Add_8*(
_output_shapes
:€€€€€€€€€А*
T0
Ы
rnn/rnn/basic_lstm_cell/Mul_12Mulrnn/rnn/basic_lstm_cell/Add_7"rnn/rnn/basic_lstm_cell/Sigmoid_12*(
_output_shapes
:€€€€€€€€€А*
T0
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_13Sigmoidrnn/rnn/basic_lstm_cell/split_4*
T0*(
_output_shapes
:€€€€€€€€€А
|
rnn/rnn/basic_lstm_cell/Tanh_8Tanh!rnn/rnn/basic_lstm_cell/split_4:1*(
_output_shapes
:€€€€€€€€€А*
T0
Ь
rnn/rnn/basic_lstm_cell/Mul_13Mul"rnn/rnn/basic_lstm_cell/Sigmoid_13rnn/rnn/basic_lstm_cell/Tanh_8*(
_output_shapes
:€€€€€€€€€А*
T0
Ч
rnn/rnn/basic_lstm_cell/Add_9Addrnn/rnn/basic_lstm_cell/Mul_12rnn/rnn/basic_lstm_cell/Mul_13*
T0*(
_output_shapes
:€€€€€€€€€А
x
rnn/rnn/basic_lstm_cell/Tanh_9Tanhrnn/rnn/basic_lstm_cell/Add_9*(
_output_shapes
:€€€€€€€€€А*
T0
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_14Sigmoid!rnn/rnn/basic_lstm_cell/split_4:3*(
_output_shapes
:€€€€€€€€€А*
T0
Ь
rnn/rnn/basic_lstm_cell/Mul_14Mulrnn/rnn/basic_lstm_cell/Tanh_9"rnn/rnn/basic_lstm_cell/Sigmoid_14*
T0*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_15Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_5/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
 rnn/rnn/basic_lstm_cell/concat_5ConcatV2input/unstack:5rnn/rnn/basic_lstm_cell/Mul_14%rnn/rnn/basic_lstm_cell/concat_5/axis*
N*(
_output_shapes
:€€€€€€€€€В*

Tidx0*
T0
∆
 rnn/rnn/basic_lstm_cell/MatMul_5MatMul rnn/rnn/basic_lstm_cell/concat_5rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( 
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_5BiasAdd rnn/rnn/basic_lstm_cell/MatMul_5rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_16Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/rnn/basic_lstm_cell/split_5Split rnn/rnn/basic_lstm_cell/Const_15!rnn/rnn/basic_lstm_cell/BiasAdd_5*
T0*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А
e
 rnn/rnn/basic_lstm_cell/Const_17Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
rnn/rnn/basic_lstm_cell/Add_10Add!rnn/rnn/basic_lstm_cell/split_5:2 rnn/rnn/basic_lstm_cell/Const_17*
T0*(
_output_shapes
:€€€€€€€€€А
А
"rnn/rnn/basic_lstm_cell/Sigmoid_15Sigmoidrnn/rnn/basic_lstm_cell/Add_10*
T0*(
_output_shapes
:€€€€€€€€€А
Ы
rnn/rnn/basic_lstm_cell/Mul_15Mulrnn/rnn/basic_lstm_cell/Add_9"rnn/rnn/basic_lstm_cell/Sigmoid_15*(
_output_shapes
:€€€€€€€€€А*
T0
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_16Sigmoidrnn/rnn/basic_lstm_cell/split_5*(
_output_shapes
:€€€€€€€€€А*
T0
}
rnn/rnn/basic_lstm_cell/Tanh_10Tanh!rnn/rnn/basic_lstm_cell/split_5:1*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_16Mul"rnn/rnn/basic_lstm_cell/Sigmoid_16rnn/rnn/basic_lstm_cell/Tanh_10*(
_output_shapes
:€€€€€€€€€А*
T0
Ш
rnn/rnn/basic_lstm_cell/Add_11Addrnn/rnn/basic_lstm_cell/Mul_15rnn/rnn/basic_lstm_cell/Mul_16*
T0*(
_output_shapes
:€€€€€€€€€А
z
rnn/rnn/basic_lstm_cell/Tanh_11Tanhrnn/rnn/basic_lstm_cell/Add_11*
T0*(
_output_shapes
:€€€€€€€€€А
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_17Sigmoid!rnn/rnn/basic_lstm_cell/split_5:3*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_17Mulrnn/rnn/basic_lstm_cell/Tanh_11"rnn/rnn/basic_lstm_cell/Sigmoid_17*
T0*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_18Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_6/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
 rnn/rnn/basic_lstm_cell/concat_6ConcatV2input/unstack:6rnn/rnn/basic_lstm_cell/Mul_17%rnn/rnn/basic_lstm_cell/concat_6/axis*(
_output_shapes
:€€€€€€€€€В*

Tidx0*
T0*
N
∆
 rnn/rnn/basic_lstm_cell/MatMul_6MatMul rnn/rnn/basic_lstm_cell/concat_6rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( 
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_6BiasAdd rnn/rnn/basic_lstm_cell/MatMul_6rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_19Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/rnn/basic_lstm_cell/split_6Split rnn/rnn/basic_lstm_cell/Const_18!rnn/rnn/basic_lstm_cell/BiasAdd_6*
T0*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А
e
 rnn/rnn/basic_lstm_cell/Const_20Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
rnn/rnn/basic_lstm_cell/Add_12Add!rnn/rnn/basic_lstm_cell/split_6:2 rnn/rnn/basic_lstm_cell/Const_20*
T0*(
_output_shapes
:€€€€€€€€€А
А
"rnn/rnn/basic_lstm_cell/Sigmoid_18Sigmoidrnn/rnn/basic_lstm_cell/Add_12*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_18Mulrnn/rnn/basic_lstm_cell/Add_11"rnn/rnn/basic_lstm_cell/Sigmoid_18*
T0*(
_output_shapes
:€€€€€€€€€А
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_19Sigmoidrnn/rnn/basic_lstm_cell/split_6*(
_output_shapes
:€€€€€€€€€А*
T0
}
rnn/rnn/basic_lstm_cell/Tanh_12Tanh!rnn/rnn/basic_lstm_cell/split_6:1*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_19Mul"rnn/rnn/basic_lstm_cell/Sigmoid_19rnn/rnn/basic_lstm_cell/Tanh_12*
T0*(
_output_shapes
:€€€€€€€€€А
Ш
rnn/rnn/basic_lstm_cell/Add_13Addrnn/rnn/basic_lstm_cell/Mul_18rnn/rnn/basic_lstm_cell/Mul_19*(
_output_shapes
:€€€€€€€€€А*
T0
z
rnn/rnn/basic_lstm_cell/Tanh_13Tanhrnn/rnn/basic_lstm_cell/Add_13*(
_output_shapes
:€€€€€€€€€А*
T0
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_20Sigmoid!rnn/rnn/basic_lstm_cell/split_6:3*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_20Mulrnn/rnn/basic_lstm_cell/Tanh_13"rnn/rnn/basic_lstm_cell/Sigmoid_20*(
_output_shapes
:€€€€€€€€€А*
T0
b
 rnn/rnn/basic_lstm_cell/Const_21Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_7/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
 rnn/rnn/basic_lstm_cell/concat_7ConcatV2input/unstack:7rnn/rnn/basic_lstm_cell/Mul_20%rnn/rnn/basic_lstm_cell/concat_7/axis*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€В
∆
 rnn/rnn/basic_lstm_cell/MatMul_7MatMul rnn/rnn/basic_lstm_cell/concat_7rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( 
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_7BiasAdd rnn/rnn/basic_lstm_cell/MatMul_7rnn/basic_lstm_cell/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
b
 rnn/rnn/basic_lstm_cell/Const_22Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/rnn/basic_lstm_cell/split_7Split rnn/rnn/basic_lstm_cell/Const_21!rnn/rnn/basic_lstm_cell/BiasAdd_7*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
e
 rnn/rnn/basic_lstm_cell/Const_23Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
rnn/rnn/basic_lstm_cell/Add_14Add!rnn/rnn/basic_lstm_cell/split_7:2 rnn/rnn/basic_lstm_cell/Const_23*
T0*(
_output_shapes
:€€€€€€€€€А
А
"rnn/rnn/basic_lstm_cell/Sigmoid_21Sigmoidrnn/rnn/basic_lstm_cell/Add_14*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_21Mulrnn/rnn/basic_lstm_cell/Add_13"rnn/rnn/basic_lstm_cell/Sigmoid_21*(
_output_shapes
:€€€€€€€€€А*
T0
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_22Sigmoidrnn/rnn/basic_lstm_cell/split_7*
T0*(
_output_shapes
:€€€€€€€€€А
}
rnn/rnn/basic_lstm_cell/Tanh_14Tanh!rnn/rnn/basic_lstm_cell/split_7:1*(
_output_shapes
:€€€€€€€€€А*
T0
Э
rnn/rnn/basic_lstm_cell/Mul_22Mul"rnn/rnn/basic_lstm_cell/Sigmoid_22rnn/rnn/basic_lstm_cell/Tanh_14*(
_output_shapes
:€€€€€€€€€А*
T0
Ш
rnn/rnn/basic_lstm_cell/Add_15Addrnn/rnn/basic_lstm_cell/Mul_21rnn/rnn/basic_lstm_cell/Mul_22*
T0*(
_output_shapes
:€€€€€€€€€А
z
rnn/rnn/basic_lstm_cell/Tanh_15Tanhrnn/rnn/basic_lstm_cell/Add_15*
T0*(
_output_shapes
:€€€€€€€€€А
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_23Sigmoid!rnn/rnn/basic_lstm_cell/split_7:3*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_23Mulrnn/rnn/basic_lstm_cell/Tanh_15"rnn/rnn/basic_lstm_cell/Sigmoid_23*(
_output_shapes
:€€€€€€€€€А*
T0
b
 rnn/rnn/basic_lstm_cell/Const_24Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_8/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
 rnn/rnn/basic_lstm_cell/concat_8ConcatV2input/unstack:8rnn/rnn/basic_lstm_cell/Mul_23%rnn/rnn/basic_lstm_cell/concat_8/axis*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€В
∆
 rnn/rnn/basic_lstm_cell/MatMul_8MatMul rnn/rnn/basic_lstm_cell/concat_8rnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( *
T0
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_8BiasAdd rnn/rnn/basic_lstm_cell/MatMul_8rnn/basic_lstm_cell/bias/read*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А*
T0
b
 rnn/rnn/basic_lstm_cell/Const_25Const*
value	B :*
dtype0*
_output_shapes
: 
н
rnn/rnn/basic_lstm_cell/split_8Split rnn/rnn/basic_lstm_cell/Const_24!rnn/rnn/basic_lstm_cell/BiasAdd_8*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
e
 rnn/rnn/basic_lstm_cell/Const_26Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
rnn/rnn/basic_lstm_cell/Add_16Add!rnn/rnn/basic_lstm_cell/split_8:2 rnn/rnn/basic_lstm_cell/Const_26*(
_output_shapes
:€€€€€€€€€А*
T0
А
"rnn/rnn/basic_lstm_cell/Sigmoid_24Sigmoidrnn/rnn/basic_lstm_cell/Add_16*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_24Mulrnn/rnn/basic_lstm_cell/Add_15"rnn/rnn/basic_lstm_cell/Sigmoid_24*(
_output_shapes
:€€€€€€€€€А*
T0
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_25Sigmoidrnn/rnn/basic_lstm_cell/split_8*(
_output_shapes
:€€€€€€€€€А*
T0
}
rnn/rnn/basic_lstm_cell/Tanh_16Tanh!rnn/rnn/basic_lstm_cell/split_8:1*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_25Mul"rnn/rnn/basic_lstm_cell/Sigmoid_25rnn/rnn/basic_lstm_cell/Tanh_16*
T0*(
_output_shapes
:€€€€€€€€€А
Ш
rnn/rnn/basic_lstm_cell/Add_17Addrnn/rnn/basic_lstm_cell/Mul_24rnn/rnn/basic_lstm_cell/Mul_25*
T0*(
_output_shapes
:€€€€€€€€€А
z
rnn/rnn/basic_lstm_cell/Tanh_17Tanhrnn/rnn/basic_lstm_cell/Add_17*(
_output_shapes
:€€€€€€€€€А*
T0
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_26Sigmoid!rnn/rnn/basic_lstm_cell/split_8:3*(
_output_shapes
:€€€€€€€€€А*
T0
Э
rnn/rnn/basic_lstm_cell/Mul_26Mulrnn/rnn/basic_lstm_cell/Tanh_17"rnn/rnn/basic_lstm_cell/Sigmoid_26*
T0*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_27Const*
value	B :*
dtype0*
_output_shapes
: 
g
%rnn/rnn/basic_lstm_cell/concat_9/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ћ
 rnn/rnn/basic_lstm_cell/concat_9ConcatV2input/unstack:9rnn/rnn/basic_lstm_cell/Mul_26%rnn/rnn/basic_lstm_cell/concat_9/axis*
T0*
N*(
_output_shapes
:€€€€€€€€€В*

Tidx0
∆
 rnn/rnn/basic_lstm_cell/MatMul_9MatMul rnn/rnn/basic_lstm_cell/concat_9rnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b( 
Ј
!rnn/rnn/basic_lstm_cell/BiasAdd_9BiasAdd rnn/rnn/basic_lstm_cell/MatMul_9rnn/basic_lstm_cell/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
b
 rnn/rnn/basic_lstm_cell/Const_28Const*
_output_shapes
: *
value	B :*
dtype0
н
rnn/rnn/basic_lstm_cell/split_9Split rnn/rnn/basic_lstm_cell/Const_27!rnn/rnn/basic_lstm_cell/BiasAdd_9*
	num_split*d
_output_shapesR
P:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А:€€€€€€€€€А*
T0
e
 rnn/rnn/basic_lstm_cell/Const_29Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Э
rnn/rnn/basic_lstm_cell/Add_18Add!rnn/rnn/basic_lstm_cell/split_9:2 rnn/rnn/basic_lstm_cell/Const_29*
T0*(
_output_shapes
:€€€€€€€€€А
А
"rnn/rnn/basic_lstm_cell/Sigmoid_27Sigmoidrnn/rnn/basic_lstm_cell/Add_18*
T0*(
_output_shapes
:€€€€€€€€€А
Ь
rnn/rnn/basic_lstm_cell/Mul_27Mulrnn/rnn/basic_lstm_cell/Add_17"rnn/rnn/basic_lstm_cell/Sigmoid_27*
T0*(
_output_shapes
:€€€€€€€€€А
Б
"rnn/rnn/basic_lstm_cell/Sigmoid_28Sigmoidrnn/rnn/basic_lstm_cell/split_9*(
_output_shapes
:€€€€€€€€€А*
T0
}
rnn/rnn/basic_lstm_cell/Tanh_18Tanh!rnn/rnn/basic_lstm_cell/split_9:1*(
_output_shapes
:€€€€€€€€€А*
T0
Э
rnn/rnn/basic_lstm_cell/Mul_28Mul"rnn/rnn/basic_lstm_cell/Sigmoid_28rnn/rnn/basic_lstm_cell/Tanh_18*(
_output_shapes
:€€€€€€€€€А*
T0
Ш
rnn/rnn/basic_lstm_cell/Add_19Addrnn/rnn/basic_lstm_cell/Mul_27rnn/rnn/basic_lstm_cell/Mul_28*
T0*(
_output_shapes
:€€€€€€€€€А
z
rnn/rnn/basic_lstm_cell/Tanh_19Tanhrnn/rnn/basic_lstm_cell/Add_19*(
_output_shapes
:€€€€€€€€€А*
T0
Г
"rnn/rnn/basic_lstm_cell/Sigmoid_29Sigmoid!rnn/rnn/basic_lstm_cell/split_9:3*
T0*(
_output_shapes
:€€€€€€€€€А
Э
rnn/rnn/basic_lstm_cell/Mul_29Mulrnn/rnn/basic_lstm_cell/Tanh_19"rnn/rnn/basic_lstm_cell/Sigmoid_29*
T0*(
_output_shapes
:€€€€€€€€€А
[
rnn/outputs/tagConst*
valueB Brnn/outputs*
dtype0*
_output_shapes
: 
£
rnn/outputs/valuesPackrnn/rnn/basic_lstm_cell/Mul_2rnn/rnn/basic_lstm_cell/Mul_5rnn/rnn/basic_lstm_cell/Mul_8rnn/rnn/basic_lstm_cell/Mul_11rnn/rnn/basic_lstm_cell/Mul_14rnn/rnn/basic_lstm_cell/Mul_17rnn/rnn/basic_lstm_cell/Mul_20rnn/rnn/basic_lstm_cell/Mul_23rnn/rnn/basic_lstm_cell/Mul_26rnn/rnn/basic_lstm_cell/Mul_29*,
_output_shapes
:
€€€€€€€€€А*
T0*

axis *
N

e
rnn/outputsHistogramSummaryrnn/outputs/tagrnn/outputs/values*
_output_shapes
: *
T0
Y
rnn/states/tagConst*
_output_shapes
: *
valueB B
rnn/states*
dtype0
•
rnn/states/valuesPackrnn/rnn/basic_lstm_cell/Add_19rnn/rnn/basic_lstm_cell/Mul_29*
T0*

axis *
N*,
_output_shapes
:€€€€€€€€€А
b

rnn/statesHistogramSummaryrnn/states/tagrnn/states/values*
T0*
_output_shapes
: 
c
rnn/rnn_weights/tagConst* 
valueB Brnn/rnn_weights*
dtype0*
_output_shapes
: 
z
rnn/rnn_weightsHistogramSummaryrnn/rnn_weights/tagrnn/basic_lstm_cell/kernel/read*
T0*
_output_shapes
: 
a
rnn/rnn_biases/tagConst*
valueB Brnn/rnn_biases*
dtype0*
_output_shapes
: 
v
rnn/rnn_biasesHistogramSummaryrnn/rnn_biases/tagrnn/basic_lstm_cell/bias/read*
_output_shapes
: *
T0
Ы
+nn1/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@nn1/kernel*
valueB"       *
dtype0*
_output_shapes
:
Н
)nn1/kernel/Initializer/random_uniform/minConst*
_class
loc:@nn1/kernel*
valueB
 *:ЌЊ*
dtype0*
_output_shapes
: 
Н
)nn1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
_class
loc:@nn1/kernel*
valueB
 *:Ќ>*
dtype0
а
3nn1/kernel/Initializer/random_uniform/RandomUniformRandomUniform+nn1/kernel/Initializer/random_uniform/shape*
_class
loc:@nn1/kernel*
seed2 *
dtype0*
_output_shapes
:	А *

seed *
T0
∆
)nn1/kernel/Initializer/random_uniform/subSub)nn1/kernel/Initializer/random_uniform/max)nn1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@nn1/kernel
ў
)nn1/kernel/Initializer/random_uniform/mulMul3nn1/kernel/Initializer/random_uniform/RandomUniform)nn1/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@nn1/kernel*
_output_shapes
:	А 
Ћ
%nn1/kernel/Initializer/random_uniformAdd)nn1/kernel/Initializer/random_uniform/mul)nn1/kernel/Initializer/random_uniform/min*
_output_shapes
:	А *
T0*
_class
loc:@nn1/kernel
Я

nn1/kernel
VariableV2*
_class
loc:@nn1/kernel*
	container *
shape:	А *
dtype0*
_output_shapes
:	А *
shared_name 
ј
nn1/kernel/AssignAssign
nn1/kernel%nn1/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
p
nn1/kernel/readIdentity
nn1/kernel*
T0*
_class
loc:@nn1/kernel*
_output_shapes
:	А 
Д
nn1/bias/Initializer/zerosConst*
_class
loc:@nn1/bias*
valueB *    *
dtype0*
_output_shapes
: 
С
nn1/bias
VariableV2*
_class
loc:@nn1/bias*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
™
nn1/bias/AssignAssignnn1/biasnn1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
e
nn1/bias/readIdentitynn1/bias*
_output_shapes
: *
T0*
_class
loc:@nn1/bias
†
nn/nn1/MatMulMatMulrnn/rnn/basic_lstm_cell/Mul_29nn1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€ *
transpose_b( 
А
nn/nn1/BiasAddBiasAddnn/nn1/MatMulnn1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€ 
U
nn/nn1/TanhTanhnn/nn1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€ 
Ы
+nn2/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@nn2/kernel*
valueB"       *
dtype0*
_output_shapes
:
Н
)nn2/kernel/Initializer/random_uniform/minConst*
_class
loc:@nn2/kernel*
valueB
 *A„Њ*
dtype0*
_output_shapes
: 
Н
)nn2/kernel/Initializer/random_uniform/maxConst*
_class
loc:@nn2/kernel*
valueB
 *A„>*
dtype0*
_output_shapes
: 
я
3nn2/kernel/Initializer/random_uniform/RandomUniformRandomUniform+nn2/kernel/Initializer/random_uniform/shape*
_class
loc:@nn2/kernel*
seed2 *
dtype0*
_output_shapes

: *

seed *
T0
∆
)nn2/kernel/Initializer/random_uniform/subSub)nn2/kernel/Initializer/random_uniform/max)nn2/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@nn2/kernel*
_output_shapes
: 
Ў
)nn2/kernel/Initializer/random_uniform/mulMul3nn2/kernel/Initializer/random_uniform/RandomUniform)nn2/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@nn2/kernel*
_output_shapes

: 
 
%nn2/kernel/Initializer/random_uniformAdd)nn2/kernel/Initializer/random_uniform/mul)nn2/kernel/Initializer/random_uniform/min*
_output_shapes

: *
T0*
_class
loc:@nn2/kernel
Э

nn2/kernel
VariableV2*
dtype0*
_output_shapes

: *
shared_name *
_class
loc:@nn2/kernel*
	container *
shape
: 
њ
nn2/kernel/AssignAssign
nn2/kernel%nn2/kernel/Initializer/random_uniform*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
o
nn2/kernel/readIdentity
nn2/kernel*
T0*
_class
loc:@nn2/kernel*
_output_shapes

: 
Д
nn2/bias/Initializer/zerosConst*
_class
loc:@nn2/bias*
valueB*    *
dtype0*
_output_shapes
:
С
nn2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@nn2/bias*
	container *
shape:
™
nn2/bias/AssignAssignnn2/biasnn2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
e
nn2/bias/readIdentitynn2/bias*
T0*
_class
loc:@nn2/bias*
_output_shapes
:
Н
nn/nn2/MatMulMatMulnn/nn1/Tanhnn2/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€*
transpose_b( *
T0
А
nn/nn2/BiasAddBiasAddnn/nn2/MatMulnn2/bias/read*'
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
a
nn/nn1_weights/tagConst*
valueB Bnn/nn1_weights*
dtype0*
_output_shapes
: 
h
nn/nn1_weightsHistogramSummarynn/nn1_weights/tagnn1/kernel/read*
T0*
_output_shapes
: 
_
nn/nn1_biases/tagConst*
valueB Bnn/nn1_biases*
dtype0*
_output_shapes
: 
d
nn/nn1_biasesHistogramSummarynn/nn1_biases/tagnn1/bias/read*
T0*
_output_shapes
: 
a
nn/nn2_weights/tagConst*
_output_shapes
: *
valueB Bnn/nn2_weights*
dtype0
h
nn/nn2_weightsHistogramSummarynn/nn2_weights/tagnn2/kernel/read*
T0*
_output_shapes
: 
_
nn/nn2_biases/tagConst*
valueB Bnn/nn2_biases*
dtype0*
_output_shapes
: 
d
nn/nn2_biasesHistogramSummarynn/nn2_biases/tagnn2/bias/read*
_output_shapes
: *
T0
k
nn/train_prediction/tagConst*$
valueB Bnn/train_prediction*
dtype0*
_output_shapes
: 
q
nn/train_predictionHistogramSummarynn/train_prediction/tagnn/nn2/BiasAdd*
_output_shapes
: *
T0
T
loss/subSubnn/nn2/BiasAddY*'
_output_shapes
:€€€€€€€€€*
T0
Q
loss/SquareSquareloss/sub*'
_output_shapes
:€€€€€€€€€*
T0
[

loss/ConstConst*
_output_shapes
:*
valueB"       *
dtype0
h
	loss/lossMeanloss/Square
loss/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
d
loss/train_loss/tagsConst*
_output_shapes
: * 
valueB Bloss/train_loss*
dtype0
b
loss/train_lossScalarSummaryloss/train_loss/tags	loss/loss*
_output_shapes
: *
T0
`
loss/val_loss/tagsConst*
_output_shapes
: *
valueB Bloss/val_loss*
dtype0
^
loss/val_lossScalarSummaryloss/val_loss/tags	loss/loss*
_output_shapes
: *
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Б
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
}
,train/gradients/loss/loss_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
ђ
&train/gradients/loss/loss_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/loss/loss_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
o
$train/gradients/loss/loss_grad/ShapeShapeloss/Square*
T0*
out_type0*
_output_shapes
:
љ
#train/gradients/loss/loss_grad/TileTile&train/gradients/loss/loss_grad/Reshape$train/gradients/loss/loss_grad/Shape*'
_output_shapes
:€€€€€€€€€*

Tmultiples0*
T0
q
&train/gradients/loss/loss_grad/Shape_1Shapeloss/Square*
out_type0*
_output_shapes
:*
T0
i
&train/gradients/loss/loss_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
n
$train/gradients/loss/loss_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ј
#train/gradients/loss/loss_grad/ProdProd&train/gradients/loss/loss_grad/Shape_1$train/gradients/loss/loss_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
p
&train/gradients/loss/loss_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
ї
%train/gradients/loss/loss_grad/Prod_1Prod&train/gradients/loss/loss_grad/Shape_2&train/gradients/loss/loss_grad/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
j
(train/gradients/loss/loss_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
£
&train/gradients/loss/loss_grad/MaximumMaximum%train/gradients/loss/loss_grad/Prod_1(train/gradients/loss/loss_grad/Maximum/y*
_output_shapes
: *
T0
°
'train/gradients/loss/loss_grad/floordivFloorDiv#train/gradients/loss/loss_grad/Prod&train/gradients/loss/loss_grad/Maximum*
T0*
_output_shapes
: 
Ф
#train/gradients/loss/loss_grad/CastCast'train/gradients/loss/loss_grad/floordiv*

SrcT0*
Truncate( *

DstT0*
_output_shapes
: 
≠
&train/gradients/loss/loss_grad/truedivRealDiv#train/gradients/loss/loss_grad/Tile#train/gradients/loss/loss_grad/Cast*
T0*'
_output_shapes
:€€€€€€€€€
Ф
&train/gradients/loss/Square_grad/ConstConst'^train/gradients/loss/loss_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
П
$train/gradients/loss/Square_grad/MulMulloss/sub&train/gradients/loss/Square_grad/Const*
T0*'
_output_shapes
:€€€€€€€€€
≠
&train/gradients/loss/Square_grad/Mul_1Mul&train/gradients/loss/loss_grad/truediv$train/gradients/loss/Square_grad/Mul*'
_output_shapes
:€€€€€€€€€*
T0
q
#train/gradients/loss/sub_grad/ShapeShapenn/nn2/BiasAdd*
_output_shapes
:*
T0*
out_type0
f
%train/gradients/loss/sub_grad/Shape_1ShapeY*
out_type0*
_output_shapes
:*
T0
’
3train/gradients/loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs#train/gradients/loss/sub_grad/Shape%train/gradients/loss/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≈
!train/gradients/loss/sub_grad/SumSum&train/gradients/loss/Square_grad/Mul_13train/gradients/loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Є
%train/gradients/loss/sub_grad/ReshapeReshape!train/gradients/loss/sub_grad/Sum#train/gradients/loss/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
…
#train/gradients/loss/sub_grad/Sum_1Sum&train/gradients/loss/Square_grad/Mul_15train/gradients/loss/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
p
!train/gradients/loss/sub_grad/NegNeg#train/gradients/loss/sub_grad/Sum_1*
T0*
_output_shapes
:
Љ
'train/gradients/loss/sub_grad/Reshape_1Reshape!train/gradients/loss/sub_grad/Neg%train/gradients/loss/sub_grad/Shape_1*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
И
.train/gradients/loss/sub_grad/tuple/group_depsNoOp&^train/gradients/loss/sub_grad/Reshape(^train/gradients/loss/sub_grad/Reshape_1
Ж
6train/gradients/loss/sub_grad/tuple/control_dependencyIdentity%train/gradients/loss/sub_grad/Reshape/^train/gradients/loss/sub_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
М
8train/gradients/loss/sub_grad/tuple/control_dependency_1Identity'train/gradients/loss/sub_grad/Reshape_1/^train/gradients/loss/sub_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€*
T0*:
_class0
.,loc:@train/gradients/loss/sub_grad/Reshape_1
≤
/train/gradients/nn/nn2/BiasAdd_grad/BiasAddGradBiasAddGrad6train/gradients/loss/sub_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
І
4train/gradients/nn/nn2/BiasAdd_grad/tuple/group_depsNoOp7^train/gradients/loss/sub_grad/tuple/control_dependency0^train/gradients/nn/nn2/BiasAdd_grad/BiasAddGrad
£
<train/gradients/nn/nn2/BiasAdd_grad/tuple/control_dependencyIdentity6train/gradients/loss/sub_grad/tuple/control_dependency5^train/gradients/nn/nn2/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@train/gradients/loss/sub_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ы
>train/gradients/nn/nn2/BiasAdd_grad/tuple/control_dependency_1Identity/train/gradients/nn/nn2/BiasAdd_grad/BiasAddGrad5^train/gradients/nn/nn2/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/nn2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Џ
)train/gradients/nn/nn2/MatMul_grad/MatMulMatMul<train/gradients/nn/nn2/BiasAdd_grad/tuple/control_dependencynn2/kernel/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€ *
transpose_b(*
T0
ѕ
+train/gradients/nn/nn2/MatMul_grad/MatMul_1MatMulnn/nn1/Tanh<train/gradients/nn/nn2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

: *
transpose_b( *
T0
Х
3train/gradients/nn/nn2/MatMul_grad/tuple/group_depsNoOp*^train/gradients/nn/nn2/MatMul_grad/MatMul,^train/gradients/nn/nn2/MatMul_grad/MatMul_1
Ш
;train/gradients/nn/nn2/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/nn/nn2/MatMul_grad/MatMul4^train/gradients/nn/nn2/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@train/gradients/nn/nn2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€ *
T0
Х
=train/gradients/nn/nn2/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/nn/nn2/MatMul_grad/MatMul_14^train/gradients/nn/nn2/MatMul_grad/tuple/group_deps*
_output_shapes

: *
T0*>
_class4
20loc:@train/gradients/nn/nn2/MatMul_grad/MatMul_1
±
)train/gradients/nn/nn1/Tanh_grad/TanhGradTanhGradnn/nn1/Tanh;train/gradients/nn/nn2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€ 
•
/train/gradients/nn/nn1/BiasAdd_grad/BiasAddGradBiasAddGrad)train/gradients/nn/nn1/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ъ
4train/gradients/nn/nn1/BiasAdd_grad/tuple/group_depsNoOp0^train/gradients/nn/nn1/BiasAdd_grad/BiasAddGrad*^train/gradients/nn/nn1/Tanh_grad/TanhGrad
Ъ
<train/gradients/nn/nn1/BiasAdd_grad/tuple/control_dependencyIdentity)train/gradients/nn/nn1/Tanh_grad/TanhGrad5^train/gradients/nn/nn1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/nn/nn1/Tanh_grad/TanhGrad*'
_output_shapes
:€€€€€€€€€ 
Ы
>train/gradients/nn/nn1/BiasAdd_grad/tuple/control_dependency_1Identity/train/gradients/nn/nn1/BiasAdd_grad/BiasAddGrad5^train/gradients/nn/nn1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/nn/nn1/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
џ
)train/gradients/nn/nn1/MatMul_grad/MatMulMatMul<train/gradients/nn/nn1/BiasAdd_grad/tuple/control_dependencynn1/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€А*
transpose_b(*
T0
г
+train/gradients/nn/nn1/MatMul_grad/MatMul_1MatMulrnn/rnn/basic_lstm_cell/Mul_29<train/gradients/nn/nn1/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	А *
transpose_b( 
Х
3train/gradients/nn/nn1/MatMul_grad/tuple/group_depsNoOp*^train/gradients/nn/nn1/MatMul_grad/MatMul,^train/gradients/nn/nn1/MatMul_grad/MatMul_1
Щ
;train/gradients/nn/nn1/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/nn/nn1/MatMul_grad/MatMul4^train/gradients/nn/nn1/MatMul_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*<
_class2
0.loc:@train/gradients/nn/nn1/MatMul_grad/MatMul
Ц
=train/gradients/nn/nn1/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/nn/nn1/MatMul_grad/MatMul_14^train/gradients/nn/nn1/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/nn/nn1/MatMul_grad/MatMul_1*
_output_shapes
:	А 
Ш
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_19*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_29*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
“
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/MulMul;train/gradients/nn/nn1/MatMul_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_29*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
—
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_19;train/gradients/nn/nn1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/Reshape_1
л
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_19_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_19Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_29Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_29_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_27*
T0*
out_type0*
_output_shapes
:
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_28*
out_type0*
_output_shapes
:*
T0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
И
7train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/SumSum=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_19_grad/TanhGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
М
9train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Sum_1Sum=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_19_grad/TanhGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/Reshape_1
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_17*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_27*
out_type0*
_output_shapes
:*
T0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
г
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/MulMulLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_27*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
б
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_17Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_28*
T0*
out_type0*
_output_shapes
:
Ъ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_18*
out_type0*
_output_shapes
:*
T0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/MulMulNtrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_18*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
з
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_28Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_19_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_27Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_28Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
н
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_18Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_28_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
9train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_9:2*
T0*
out_type0*
_output_shapes
:
~
;train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
7train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Т
9train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_27_grad/SigmoidGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
п
=train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
”
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/Reshape_1*
_output_shapes
: 
»
;train/gradients/rnn/rnn/basic_lstm_cell/split_9_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_28_grad/SigmoidGrad=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_18_grad/TanhGradLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_18_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_29_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_27*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_9_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_9_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_9_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_9_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad*
_output_shapes	
:А
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_9Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
ВА
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1* 
_output_shapes
:
ВА
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_9/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Rank*
_output_shapes
: *
T0
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeShapeinput/unstack:9*
T0*
out_type0*
_output_shapes
:
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeNShapeNinput/unstack:9rnn/rnn/basic_lstm_cell/Mul_26*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/Slice_1
Ш
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_17*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_26*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_26*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_17Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_9_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/Reshape_1
л
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_17_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_17Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_26Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_26_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
µ
train/gradients/AddNAddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/tuple/control_dependency=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_17_grad/TanhGrad*
N*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_27_grad/Reshape
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_24*
T0*
out_type0*
_output_shapes
:
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_25*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
7train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/SumSumtrain/gradients/AddNItrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
г
9train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Sum_1Sumtrain/gradients/AddNKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/Reshape_1
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_15*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_24*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
г
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/MulMulLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_24*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
б
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_15Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_25*
_output_shapes
:*
T0*
out_type0
Ъ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_16*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/MulMulNtrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_16*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
з
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_25Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_17_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/Reshape_1
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_24Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_25Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
н
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_16Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_25_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ъ
9train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_8:2*
_output_shapes
:*
T0*
out_type0
~
;train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
7train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
Т
9train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_24_grad/SigmoidGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
п
=train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
”
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/Reshape_1*
_output_shapes
: 
»
;train/gradients/rnn/rnn/basic_lstm_cell/split_8_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_25_grad/SigmoidGrad=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_16_grad/TanhGradLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_16_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_26_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_24*
N*(
_output_shapes
:€€€€€€€€€А*

Tidx0*
T0
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_8_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_8_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_8_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_8_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/group_deps*
_output_shapes	
:А*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/BiasAddGrad
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_8Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/group_deps* 
_output_shapes
:
ВА*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/MatMul_1
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_8/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Rank*
_output_shapes
: *
T0
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeShapeinput/unstack:8*
T0*
out_type0*
_output_shapes
:
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeNShapeNinput/unstack:8rnn/rnn/basic_lstm_cell/Mul_23*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ш
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_15*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_23*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_23*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_15Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_8_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
л
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_15_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_15Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_23Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_23_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ј
train/gradients/AddN_1AddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/tuple/control_dependency=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_15_grad/TanhGrad*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_24_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_21*
T0*
out_type0*
_output_shapes
:
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_22*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
б
7train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/SumSumtrain/gradients/AddN_1Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
е
9train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Sum_1Sumtrain/gradients/AddN_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_13*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_21*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/MulMulLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_21*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
б
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_13Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_22*
T0*
out_type0*
_output_shapes
:
Ъ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_14*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/MulMulNtrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_14*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
з
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_22Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_15_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/Reshape_1
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_21Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_22Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
н
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_14Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_22_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
9train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_7:2*
_output_shapes
:*
T0*
out_type0
~
;train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
7train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Т
9train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_21_grad/SigmoidGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
п
=train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
”
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/group_deps*
_output_shapes
: *
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/Reshape_1
»
;train/gradients/rnn/rnn/basic_lstm_cell/split_7_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_22_grad/SigmoidGrad=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_14_grad/TanhGradLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_14_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_23_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_21*
T0*
N*(
_output_shapes
:€€€€€€€€€А*

Tidx0
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_7_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_7_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_7_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_7_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/BiasAddGrad*
_output_shapes	
:А
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_7Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/MatMul_1* 
_output_shapes
:
ВА
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_7/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Rank*
T0*
_output_shapes
: 
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeShapeinput/unstack:7*
T0*
out_type0*
_output_shapes
:
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeNShapeNinput/unstack:7rnn/rnn/basic_lstm_cell/Mul_20*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeN*'
_output_shapes
:€€€€€€€€€*
T0*
Index0
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А
Ш
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_13*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_20*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_20*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_13Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_7_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/Reshape_1
л
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_13_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_13Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_20Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_20_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ј
train/gradients/AddN_2AddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/tuple/control_dependency=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_13_grad/TanhGrad*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_21_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_18*
T0*
out_type0*
_output_shapes
:
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_19*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
б
7train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/SumSumtrain/gradients/AddN_2Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
е
9train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Sum_1Sumtrain/gradients/AddN_2Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/Reshape_1
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_11*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_18*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
г
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/MulMulLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_18*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
б
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_11Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_19*
_output_shapes
:*
T0*
out_type0
Ъ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_12*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/MulMulNtrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_12*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
з
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_19Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_13_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_18Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_19Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
н
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_12Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_19_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
9train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_6:2*
T0*
out_type0*
_output_shapes
:
~
;train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
7train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Т
9train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_18_grad/SigmoidGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
п
=train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
”
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/Reshape_1*
_output_shapes
: 
»
;train/gradients/rnn/rnn/basic_lstm_cell/split_6_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_19_grad/SigmoidGrad=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_12_grad/TanhGradLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_12_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_20_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_18*
T0*
N*(
_output_shapes
:€€€€€€€€€А*

Tidx0
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_6_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_6_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_6_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_6_grad/concat
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/BiasAddGrad*
_output_shapes	
:А
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_6Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
ВА
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/MatMul_1* 
_output_shapes
:
ВА
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_6/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Rank*
T0*
_output_shapes
: 
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeShapeinput/unstack:6*
_output_shapes
:*
T0*
out_type0
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeNShapeNinput/unstack:6rnn/rnn/basic_lstm_cell/Mul_17*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ш
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_11*
out_type0*
_output_shapes
:*
T0
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_17*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_17*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_11Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_6_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
л
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_11_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_11Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_17Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_17_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ј
train/gradients/AddN_3AddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/tuple/control_dependency=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_11_grad/TanhGrad*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_18_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_15*
_output_shapes
:*
T0*
out_type0
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_16*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
б
7train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/SumSumtrain/gradients/AddN_3Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
е
9train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Sum_1Sumtrain/gradients/AddN_3Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ц
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_9*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_15*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
г
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/MulMulLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_15*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
а
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_9Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_16*
out_type0*
_output_shapes
:*
T0
Ъ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_10*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/MulMulNtrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_10*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
з
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_16Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_11_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_15Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_16Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
н
=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_10Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_16_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Ъ
9train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_5:2*
T0*
out_type0*
_output_shapes
:
~
;train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
О
7train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Т
9train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_15_grad/SigmoidGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
п
=train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape
”
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/Reshape_1*
_output_shapes
: 
»
;train/gradients/rnn/rnn/basic_lstm_cell/split_5_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_16_grad/SigmoidGrad=train/gradients/rnn/rnn/basic_lstm_cell/Tanh_10_grad/TanhGradLtrain/gradients/rnn/rnn/basic_lstm_cell/Add_10_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_17_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_15*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_5_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_5_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_5_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_5_grad/concat
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/group_deps*
_output_shapes	
:А*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/BiasAddGrad
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(*
T0
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_5Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/MatMul_1* 
_output_shapes
:
ВА
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_5/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Rank*
T0*
_output_shapes
: 
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeShapeinput/unstack:5*
T0*
out_type0*
_output_shapes
:
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeNShapeNinput/unstack:5rnn/rnn/basic_lstm_cell/Mul_14*
N* 
_output_shapes
::*
T0*
out_type0
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_9*
_output_shapes
:*
T0*
out_type0
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_14*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_14*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
е
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_9Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_5_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
й
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_9_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_9Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_14Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_14_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ґ
train/gradients/AddN_4AddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/tuple/control_dependency<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_9_grad/TanhGrad*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_15_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Ц
8train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_12*
T0*
out_type0*
_output_shapes
:
Ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_13*
out_type0*
_output_shapes
:*
T0
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
я
6train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/SumSumtrain/gradients/AddN_4Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
г
8train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Sum_1Sumtrain/gradients/AddN_4Jtrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ц
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_7*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_12*
T0*
out_type0*
_output_shapes
:
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
в
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/MulMulKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency"rnn/rnn/basic_lstm_cell/Sigmoid_12*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
я
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_7Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_13*
_output_shapes
:*
T0*
out_type0
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_8*
_output_shapes
:*
T0*
out_type0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/MulMulMtrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_8*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_13Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_9_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_12Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_13Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
л
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_8Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_13_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_4:2*
T0*
out_type0*
_output_shapes
:
}
:train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
М
6train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/SumSumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGradHtrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
Р
8train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Sum_1SumCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_12_grad/SigmoidGradJtrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
м
<train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
ѕ
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/Reshape_1*
_output_shapes
: *
T0
∆
;train/gradients/rnn/rnn/basic_lstm_cell/split_4_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_13_grad/SigmoidGrad<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_8_grad/TanhGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_8_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_14_grad/SigmoidGrad rnn/rnn/basic_lstm_cell/Const_12*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_4_grad/concat*
data_formatNHWC*
_output_shapes	
:А*
T0
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_4_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_4_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_4_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/BiasAddGrad*
_output_shapes	
:А*
T0
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(*
T0
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_4Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul*(
_output_shapes
:€€€€€€€€€В*
T0
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/group_deps*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/MatMul_1* 
_output_shapes
:
ВА*
T0
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_4/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeShapeinput/unstack:4*
out_type0*
_output_shapes
:*
T0
ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeNShapeNinput/unstack:4rnn/rnn/basic_lstm_cell/Mul_11*
out_type0*
N* 
_output_shapes
::*
T0
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/ShapeN:1*(
_output_shapes
:€€€€€€€€€А*
T0*
Index0
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice*'
_output_shapes
:€€€€€€€€€*
T0
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А
Ч
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_7*
T0*
out_type0*
_output_shapes
:
Э
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Shape_1Shape"rnn/rnn/basic_lstm_cell/Sigmoid_11*
out_type0*
_output_shapes
:*
T0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
з
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1"rnn/rnn/basic_lstm_cell/Sigmoid_11*(
_output_shapes
:€€€€€€€€€А*
T0
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
е
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_7Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_4_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/group_deps*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
й
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_7_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_7Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
щ
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_11Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_11_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
ґ
train/gradients/AddN_5AddNLtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/tuple/control_dependency<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_7_grad/TanhGrad*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_12_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А*
T0
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_9*
out_type0*
_output_shapes
:*
T0
Ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_10*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
я
6train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/SumSumtrain/gradients/AddN_5Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
г
8train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Sum_1Sumtrain/gradients/AddN_5Jtrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_5*
out_type0*
_output_shapes
:*
T0
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_9*
out_type0*
_output_shapes
:*
T0
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/MulMulKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency!rnn/rnn/basic_lstm_cell/Sigmoid_9*(
_output_shapes
:€€€€€€€€€А*
T0
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ё
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_5Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ы
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/ShapeShape"rnn/rnn/basic_lstm_cell/Sigmoid_10*
T0*
out_type0*
_output_shapes
:
Щ
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_6*
out_type0*
_output_shapes
:*
T0
Ч
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgsBroadcastGradientArgs9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Shape;train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/MulMulMtrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_6*
T0*(
_output_shapes
:€€€€€€€€€А
В
7train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/SumSum7train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/MulItrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ы
;train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/ReshapeReshape7train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ж
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Mul"rnn/rnn/basic_lstm_cell/Sigmoid_10Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_7_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
И
9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Sum_1Sum9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Mul_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Б
=train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1Reshape9train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Sum_1;train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
 
Dtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape>^train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1
я
Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/ReshapeE^train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1E^train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/Reshape_1
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_9Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
ч
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGradSigmoidGrad"rnn/rnn/basic_lstm_cell/Sigmoid_10Ltrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
л
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_6Ntrain/gradients/rnn/rnn/basic_lstm_cell/Mul_10_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_3:2*
T0*
out_type0*
_output_shapes
:
}
:train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Л
6train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/SumSumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGradHtrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
П
8train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Sum_1SumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_9_grad/SigmoidGradJtrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
м
<train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ѕ
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/group_deps*
_output_shapes
: *
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/Reshape_1
≈
;train/gradients/rnn/rnn/basic_lstm_cell/split_3_grad/concatConcatV2Ctrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_10_grad/SigmoidGrad<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_6_grad/TanhGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_6_grad/tuple/control_dependencyCtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_11_grad/SigmoidGradrnn/rnn/basic_lstm_cell/Const_9*
T0*
N*(
_output_shapes
:€€€€€€€€€А*

Tidx0
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_3_grad/concat*
data_formatNHWC*
_output_shapes	
:А*
T0
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_3_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_3_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_3_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:А
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_3Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( *
T0
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul*(
_output_shapes
:€€€€€€€€€В*
T0
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/MatMul_1* 
_output_shapes
:
ВА
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_3/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Rank*
_output_shapes
: *
T0
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeShapeinput/unstack:3*
_output_shapes
:*
T0*
out_type0
Ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeNShapeNinput/unstack:3rnn/rnn/basic_lstm_cell/Mul_8*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeN*'
_output_shapes
:€€€€€€€€€*
T0*
Index0
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/Slice_1
Ц
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_5*
_output_shapes
:*
T0*
out_type0
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_8*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
е
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1!rnn/rnn/basic_lstm_cell/Sigmoid_8*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_5Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
и
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_5_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_5Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_8Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_8_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
і
train/gradients/AddN_6AddNKtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/tuple/control_dependency<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_5_grad/TanhGrad*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_9_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_6*
_output_shapes
:*
T0*
out_type0
Ч
:train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_7*
_output_shapes
:*
T0*
out_type0
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
6train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/SumSumtrain/gradients/AddN_6Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
г
8train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Sum_1Sumtrain/gradients/AddN_6Jtrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_3*
T0*
out_type0*
_output_shapes
:
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_6*
_output_shapes
:*
T0*
out_type0
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
а
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/MulMulKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency!rnn/rnn/basic_lstm_cell/Sigmoid_6*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ё
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_3Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/ShapeShape!rnn/rnn/basic_lstm_cell/Sigmoid_7*
_output_shapes
:*
T0*
out_type0
Ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_4*
_output_shapes
:*
T0*
out_type0
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/MulMulMtrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_4*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Mul_1Mul!rnn/rnn/basic_lstm_cell/Sigmoid_7Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_5_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/Reshape_1
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_6Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ф
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_7Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
к
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_4Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_7_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_2:2*
T0*
out_type0*
_output_shapes
:
}
:train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Л
6train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/SumSumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGradHtrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
П
8train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Sum_1SumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_6_grad/SigmoidGradJtrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
м
<train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ѕ
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/Reshape_1*
_output_shapes
: 
√
;train/gradients/rnn/rnn/basic_lstm_cell/split_2_grad/concatConcatV2Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_7_grad/SigmoidGrad<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_4_grad/TanhGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_4_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_8_grad/SigmoidGradrnn/rnn/basic_lstm_cell/Const_6*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€А
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_2_grad/concat*
data_formatNHWC*
_output_shapes	
:А*
T0
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_2_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_2_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_2_grad/concat
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/group_deps*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/BiasAddGrad*
_output_shapes	
:А*
T0
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_2Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€В*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/group_deps* 
_output_shapes
:
ВА*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/MatMul_1
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_2/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Rank*
_output_shapes
: *
T0
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeShapeinput/unstack:2*
out_type0*
_output_shapes
:*
T0
Ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeNShapeNinput/unstack:2rnn/rnn/basic_lstm_cell/Mul_5*
T0*
out_type0*
N* 
_output_shapes
::
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1* 
_output_shapes
::*
N
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeN*'
_output_shapes
:€€€€€€€€€*
T0*
Index0
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice*'
_output_shapes
:€€€€€€€€€*
T0
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/group_deps*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/Slice_1*(
_output_shapes
:€€€€€€€€€А*
T0
Ц
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_3*
_output_shapes
:*
T0*
out_type0
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_5*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
е
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1!rnn/rnn/basic_lstm_cell/Sigmoid_5*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_3Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/group_deps*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
и
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_3_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_3Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_5Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_5_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
і
train/gradients/AddN_7AddNKtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/tuple/control_dependency<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_3_grad/TanhGrad*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_6_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul_3*
out_type0*
_output_shapes
:*
T0
Ч
:train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_4*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
6train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/SumSumtrain/gradients/AddN_7Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
г
8train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Sum_1Sumtrain/gradients/AddN_7Jtrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Х
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/ShapeShapernn/rnn/basic_lstm_cell/Add_1*
T0*
out_type0*
_output_shapes
:
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_3*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
а
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/MulMulKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency!rnn/rnn/basic_lstm_cell/Sigmoid_3*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ё
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Add_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/ShapeShape!rnn/rnn/basic_lstm_cell/Sigmoid_4*
_output_shapes
:*
T0*
out_type0
Ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh_2*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
я
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/MulMulMtrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh_2*(
_output_shapes
:€€€€€€€€€А*
T0
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Mul_1Mul!rnn/rnn/basic_lstm_cell/Sigmoid_4Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape*(
_output_shapes
:€€€€€€€€€А*
T0
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_3Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ф
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_4Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency*
T0*(
_output_shapes
:€€€€€€€€€А
к
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_2Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_4_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/ShapeShape!rnn/rnn/basic_lstm_cell/split_1:2*
T0*
out_type0*
_output_shapes
:
}
:train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Л
6train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/SumSumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradHtrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
П
8train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Sum_1SumBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_3_grad/SigmoidGradJtrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
м
<train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ѕ
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/Reshape_1*
_output_shapes
: 
√
;train/gradients/rnn/rnn/basic_lstm_cell/split_1_grad/concatConcatV2Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_4_grad/SigmoidGrad<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_2_grad/TanhGradKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_2_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_5_grad/SigmoidGradrnn/rnn/basic_lstm_cell/Const_3*
N*(
_output_shapes
:€€€€€€€€€А*

Tidx0*
T0
Ћ
Btrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradBiasAddGrad;train/gradients/rnn/rnn/basic_lstm_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
“
Gtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_depsNoOpC^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad<^train/gradients/rnn/rnn/basic_lstm_cell/split_1_grad/concat
е
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/split_1_grad/concatH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_1_grad/concat*(
_output_shapes
:€€€€€€€€€А
и
Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1IdentityBtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGradH^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:А
С
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMulMatMulOtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В*
transpose_b(
М
>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1MatMul rnn/rnn/basic_lstm_cell/concat_1Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
ќ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_depsNoOp=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul?^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1
е
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependencyIdentity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMulG^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
г
Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Identity>train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1G^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/group_deps* 
_output_shapes
:
ВА*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/MatMul_1
|
:train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
…
9train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/modFloorMod%rnn/rnn/basic_lstm_cell/concat_1/axis:train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Rank*
_output_shapes
: *
T0
К
;train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeShapeinput/unstack:1*
T0*
out_type0*
_output_shapes
:
Ї
<train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeNShapeNinput/unstack:1rnn/rnn/basic_lstm_cell/Mul_2*
N* 
_output_shapes
::*
T0*
out_type0
ґ
Btrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ConcatOffsetConcatOffset9train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/mod<train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeN>train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
N* 
_output_shapes
::
’
;train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/SliceSliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset<train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeN*
T0*
Index0*'
_output_shapes
:€€€€€€€€€
№
=train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice_1SliceNtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependencyDtrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ConcatOffset:1>train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/ShapeN:1*
T0*
Index0*(
_output_shapes
:€€€€€€€€€А
ћ
Ftrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/group_depsNoOp<^train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice>^train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice_1
в
Ntrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependencyIdentity;train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/SliceG^train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice*'
_output_shapes
:€€€€€€€€€
й
Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1Identity=train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice_1G^train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/group_deps*(
_output_shapes
:€€€€€€€€€А*
T0*P
_classF
DBloc:@train/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/Slice_1
Ц
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/ShapeShapernn/rnn/basic_lstm_cell/Tanh_1*
T0*
out_type0*
_output_shapes
:
Ы
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Shape_1Shape!rnn/rnn/basic_lstm_cell/Sigmoid_2*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
е
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/MulMulPtrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1!rnn/rnn/basic_lstm_cell/Sigmoid_2*
T0*(
_output_shapes
:€€€€€€€€€А
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Mul_1Mulrnn/rnn/basic_lstm_cell/Tanh_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/concat_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
и
<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/Tanh_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
ц
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_2Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
і
train/gradients/AddN_8AddNKtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/tuple/control_dependency<train/gradients/rnn/rnn/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_3_grad/Reshape*
N*(
_output_shapes
:€€€€€€€€€А
У
8train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/ShapeShapernn/rnn/basic_lstm_cell/Mul*
T0*
out_type0*
_output_shapes
:
Ч
:train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Mul_1*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
я
6train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/SumSumtrain/gradients/AddN_8Htrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
г
8train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Sum_1Sumtrain/gradients/AddN_8Jtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Shape_1*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
Ъ
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/ShapeShape$rnn/rnn/BasicLSTMCellZeroState/zeros*
_output_shapes
:*
T0*
out_type0
Ч
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Sigmoid*
T0*
out_type0*
_output_shapes
:
О
Ftrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Shape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
№
4train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/MulMulKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependencyrnn/rnn/basic_lstm_cell/Sigmoid*(
_output_shapes
:€€€€€€€€€А*
T0
щ
4train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/SumSum4train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/MulFtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
т
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/ReshapeReshape4train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Sum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Shape*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
г
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Mul_1Mul$rnn/rnn/BasicLSTMCellZeroState/zerosKtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Sum_1Sum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Mul_1Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape_1Reshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Sum_18train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Shape_1*
Tshape0*(
_output_shapes
:€€€€€€€€€А*
T0
Ѕ
Atrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp9^train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape_1
”
Itrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity8train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/ReshapeB^train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
ў
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape_1B^train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/group_deps*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А*
T0
Щ
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/ShapeShape!rnn/rnn/basic_lstm_cell/Sigmoid_1*
T0*
out_type0*
_output_shapes
:
Ц
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Shape_1Shapernn/rnn/basic_lstm_cell/Tanh*
T0*
out_type0*
_output_shapes
:
Ф
Htrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Shape:train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
Ё
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/MulMulMtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1rnn/rnn/basic_lstm_cell/Tanh*(
_output_shapes
:€€€€€€€€€А*
T0
€
6train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/SumSum6train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/MulHtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ш
:train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/ReshapeReshape6train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Shape*(
_output_shapes
:€€€€€€€€€А*
T0*
Tshape0
д
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Mul_1Mul!rnn/rnn/basic_lstm_cell/Sigmoid_1Mtrain/gradients/rnn/rnn/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1*(
_output_shapes
:€€€€€€€€€А*
T0
Е
8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Sum_1Sum8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Mul_1Jtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ю
<train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1Reshape8train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Sum_1:train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
«
Ctrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape=^train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1
џ
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/ReshapeD^train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
б
Mtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1D^train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/Reshape_1*(
_output_shapes
:€€€€€€€€€А
р
@train/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradrnn/rnn/basic_lstm_cell/SigmoidKtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
ф
Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad!rnn/rnn/basic_lstm_cell/Sigmoid_1Ktrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*(
_output_shapes
:€€€€€€€€€А*
T0
ж
:train/gradients/rnn/rnn/basic_lstm_cell/Tanh_grad/TanhGradTanhGradrnn/rnn/basic_lstm_cell/TanhMtrain/gradients/rnn/rnn/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:€€€€€€€€€А
Х
6train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/ShapeShapernn/rnn/basic_lstm_cell/split:2*
T0*
out_type0*
_output_shapes
:
{
8train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0
О
Ftrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgs6train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Shape8train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Е
4train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/SumSum@train/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradFtrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
т
8train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/ReshapeReshape4train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Sum6train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
Й
6train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Sum_1Sum@train/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_grad/SigmoidGradHtrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ж
:train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape_1Reshape6train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Sum_18train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ѕ
Atrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/group_depsNoOp9^train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape;^train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape_1
”
Itrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity8train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/ReshapeB^train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*K
_classA
?=loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape*(
_output_shapes
:€€€€€€€€€А
«
Ktrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity:train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape_1B^train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/Add_grad/Reshape_1*
_output_shapes
: 
ї
9train/gradients/rnn/rnn/basic_lstm_cell/split_grad/concatConcatV2Btrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad:train/gradients/rnn/rnn/basic_lstm_cell/Tanh_grad/TanhGradItrain/gradients/rnn/rnn/basic_lstm_cell/Add_grad/tuple/control_dependencyBtrain/gradients/rnn/rnn/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradrnn/rnn/basic_lstm_cell/Const*

Tidx0*
T0*
N*(
_output_shapes
:€€€€€€€€€А
«
@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad9train/gradients/rnn/rnn/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:А
ћ
Etrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpA^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad:^train/gradients/rnn/rnn/basic_lstm_cell/split_grad/concat
Ё
Mtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity9train/gradients/rnn/rnn/basic_lstm_cell/split_grad/concatF^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@train/gradients/rnn/rnn/basic_lstm_cell/split_grad/concat*(
_output_shapes
:€€€€€€€€€А
а
Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1Identity@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGradF^train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
Н
:train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMulMatMulMtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyrnn/basic_lstm_cell/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:€€€€€€€€€В
Ж
<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul_1MatMulrnn/rnn/basic_lstm_cell/concatMtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
ВА*
transpose_b( 
»
Dtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp;^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul=^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul_1
Ё
Ltrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity:train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMulE^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€В
џ
Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity<train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul_1E^train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
ВА
а
train/gradients/AddN_9AddNQtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_8_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_7_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_6_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_5_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_4_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_3_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_2_grad/tuple/control_dependency_1Qtrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_1_grad/tuple/control_dependency_1Otrain/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*U
_classK
IGloc:@train/gradients/rnn/rnn/basic_lstm_cell/BiasAdd_9_grad/BiasAddGrad*
N
*
_output_shapes	
:А
Ў
train/gradients/AddN_10AddNPtrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_8_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_7_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_6_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_5_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_4_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_3_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_2_grad/tuple/control_dependency_1Ptrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_1_grad/tuple/control_dependency_1Ntrain/gradients/rnn/rnn/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0*Q
_classG
ECloc:@train/gradients/rnn/rnn/basic_lstm_cell/MatMul_9_grad/MatMul_1*
N
* 
_output_shapes
:
ВА
Б
train/beta1_power/initial_valueConst*
_class
loc:@nn1/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Т
train/beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@nn1/bias*
	container 
љ
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
s
train/beta1_power/readIdentitytrain/beta1_power*
T0*
_class
loc:@nn1/bias*
_output_shapes
: 
Б
train/beta2_power/initial_valueConst*
_class
loc:@nn1/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Т
train/beta2_power
VariableV2*
shared_name *
_class
loc:@nn1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
љ
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
s
train/beta2_power/readIdentitytrain/beta2_power*
T0*
_class
loc:@nn1/bias*
_output_shapes
: 
∆
Frnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros/shape_as_tensorConst*
valueB"     *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
∞
<rnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
∞
6rnn/basic_lstm_cell/kernel/optimizer/Initializer/zerosFillFrnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros/shape_as_tensor<rnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА
Ћ
$rnn/basic_lstm_cell/kernel/optimizer
VariableV2*
dtype0* 
_output_shapes
:
ВА*
shared_name *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
	container *
shape:
ВА
Ц
+rnn/basic_lstm_cell/kernel/optimizer/AssignAssign$rnn/basic_lstm_cell/kernel/optimizer6rnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(
µ
)rnn/basic_lstm_cell/kernel/optimizer/readIdentity$rnn/basic_lstm_cell/kernel/optimizer*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА
»
Hrnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros/shape_as_tensorConst*
valueB"     *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
:
≤
>rnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
dtype0*
_output_shapes
: 
ґ
8rnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zerosFillHrnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros/shape_as_tensor>rnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА
Ќ
&rnn/basic_lstm_cell/kernel/optimizer_1
VariableV2*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
	container *
shape:
ВА*
dtype0* 
_output_shapes
:
ВА*
shared_name 
Ь
-rnn/basic_lstm_cell/kernel/optimizer_1/AssignAssign&rnn/basic_lstm_cell/kernel/optimizer_18rnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
є
+rnn/basic_lstm_cell/kernel/optimizer_1/readIdentity&rnn/basic_lstm_cell/kernel/optimizer_1*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel* 
_output_shapes
:
ВА
Љ
Drnn/basic_lstm_cell/bias/optimizer/Initializer/zeros/shape_as_tensorConst*
valueB:А*+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
:
ђ
:rnn/basic_lstm_cell/bias/optimizer/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0
£
4rnn/basic_lstm_cell/bias/optimizer/Initializer/zerosFillDrnn/basic_lstm_cell/bias/optimizer/Initializer/zeros/shape_as_tensor:rnn/basic_lstm_cell/bias/optimizer/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes	
:А
љ
"rnn/basic_lstm_cell/bias/optimizer
VariableV2*
shared_name *+
_class!
loc:@rnn/basic_lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А
Й
)rnn/basic_lstm_cell/bias/optimizer/AssignAssign"rnn/basic_lstm_cell/bias/optimizer4rnn/basic_lstm_cell/bias/optimizer/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
™
'rnn/basic_lstm_cell/bias/optimizer/readIdentity"rnn/basic_lstm_cell/bias/optimizer*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes	
:А
Њ
Frnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:А*+
_class!
loc:@rnn/basic_lstm_cell/bias
Ѓ
<rnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@rnn/basic_lstm_cell/bias*
dtype0*
_output_shapes
: 
©
6rnn/basic_lstm_cell/bias/optimizer_1/Initializer/zerosFillFrnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros/shape_as_tensor<rnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros/Const*
_output_shapes	
:А*
T0*

index_type0*+
_class!
loc:@rnn/basic_lstm_cell/bias
њ
$rnn/basic_lstm_cell/bias/optimizer_1
VariableV2*+
_class!
loc:@rnn/basic_lstm_cell/bias*
	container *
shape:А*
dtype0*
_output_shapes	
:А*
shared_name 
П
+rnn/basic_lstm_cell/bias/optimizer_1/AssignAssign$rnn/basic_lstm_cell/bias/optimizer_16rnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
Ѓ
)rnn/basic_lstm_cell/bias/optimizer_1/readIdentity$rnn/basic_lstm_cell/bias/optimizer_1*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
_output_shapes	
:А
¶
6nn1/kernel/optimizer/Initializer/zeros/shape_as_tensorConst*
valueB"       *
_class
loc:@nn1/kernel*
dtype0*
_output_shapes
:
Р
,nn1/kernel/optimizer/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@nn1/kernel*
dtype0*
_output_shapes
: 
п
&nn1/kernel/optimizer/Initializer/zerosFill6nn1/kernel/optimizer/Initializer/zeros/shape_as_tensor,nn1/kernel/optimizer/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@nn1/kernel*
_output_shapes
:	А 
©
nn1/kernel/optimizer
VariableV2*
dtype0*
_output_shapes
:	А *
shared_name *
_class
loc:@nn1/kernel*
	container *
shape:	А 
’
nn1/kernel/optimizer/AssignAssignnn1/kernel/optimizer&nn1/kernel/optimizer/Initializer/zeros*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(
Д
nn1/kernel/optimizer/readIdentitynn1/kernel/optimizer*
_output_shapes
:	А *
T0*
_class
loc:@nn1/kernel
®
8nn1/kernel/optimizer_1/Initializer/zeros/shape_as_tensorConst*
valueB"       *
_class
loc:@nn1/kernel*
dtype0*
_output_shapes
:
Т
.nn1/kernel/optimizer_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@nn1/kernel*
dtype0*
_output_shapes
: 
х
(nn1/kernel/optimizer_1/Initializer/zerosFill8nn1/kernel/optimizer_1/Initializer/zeros/shape_as_tensor.nn1/kernel/optimizer_1/Initializer/zeros/Const*
_output_shapes
:	А *
T0*

index_type0*
_class
loc:@nn1/kernel
Ђ
nn1/kernel/optimizer_1
VariableV2*
	container *
shape:	А *
dtype0*
_output_shapes
:	А *
shared_name *
_class
loc:@nn1/kernel
џ
nn1/kernel/optimizer_1/AssignAssignnn1/kernel/optimizer_1(nn1/kernel/optimizer_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
И
nn1/kernel/optimizer_1/readIdentitynn1/kernel/optimizer_1*
T0*
_class
loc:@nn1/kernel*
_output_shapes
:	А 
О
$nn1/bias/optimizer/Initializer/zerosConst*
valueB *    *
_class
loc:@nn1/bias*
dtype0*
_output_shapes
: 
Ы
nn1/bias/optimizer
VariableV2*
shared_name *
_class
loc:@nn1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
»
nn1/bias/optimizer/AssignAssignnn1/bias/optimizer$nn1/bias/optimizer/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
y
nn1/bias/optimizer/readIdentitynn1/bias/optimizer*
T0*
_class
loc:@nn1/bias*
_output_shapes
: 
Р
&nn1/bias/optimizer_1/Initializer/zerosConst*
valueB *    *
_class
loc:@nn1/bias*
dtype0*
_output_shapes
: 
Э
nn1/bias/optimizer_1
VariableV2*
shared_name *
_class
loc:@nn1/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ќ
nn1/bias/optimizer_1/AssignAssignnn1/bias/optimizer_1&nn1/bias/optimizer_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
}
nn1/bias/optimizer_1/readIdentitynn1/bias/optimizer_1*
T0*
_class
loc:@nn1/bias*
_output_shapes
: 
Ъ
&nn2/kernel/optimizer/Initializer/zerosConst*
valueB *    *
_class
loc:@nn2/kernel*
dtype0*
_output_shapes

: 
І
nn2/kernel/optimizer
VariableV2*
dtype0*
_output_shapes

: *
shared_name *
_class
loc:@nn2/kernel*
	container *
shape
: 
‘
nn2/kernel/optimizer/AssignAssignnn2/kernel/optimizer&nn2/kernel/optimizer/Initializer/zeros*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Г
nn2/kernel/optimizer/readIdentitynn2/kernel/optimizer*
_class
loc:@nn2/kernel*
_output_shapes

: *
T0
Ь
(nn2/kernel/optimizer_1/Initializer/zerosConst*
_output_shapes

: *
valueB *    *
_class
loc:@nn2/kernel*
dtype0
©
nn2/kernel/optimizer_1
VariableV2*
dtype0*
_output_shapes

: *
shared_name *
_class
loc:@nn2/kernel*
	container *
shape
: 
Џ
nn2/kernel/optimizer_1/AssignAssignnn2/kernel/optimizer_1(nn2/kernel/optimizer_1/Initializer/zeros*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
З
nn2/kernel/optimizer_1/readIdentitynn2/kernel/optimizer_1*
_output_shapes

: *
T0*
_class
loc:@nn2/kernel
О
$nn2/bias/optimizer/Initializer/zerosConst*
valueB*    *
_class
loc:@nn2/bias*
dtype0*
_output_shapes
:
Ы
nn2/bias/optimizer
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@nn2/bias*
	container *
shape:
»
nn2/bias/optimizer/AssignAssignnn2/bias/optimizer$nn2/bias/optimizer/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
y
nn2/bias/optimizer/readIdentitynn2/bias/optimizer*
_class
loc:@nn2/bias*
_output_shapes
:*
T0
Р
&nn2/bias/optimizer_1/Initializer/zerosConst*
valueB*    *
_class
loc:@nn2/bias*
dtype0*
_output_shapes
:
Э
nn2/bias/optimizer_1
VariableV2*
shared_name *
_class
loc:@nn2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
ќ
nn2/bias/optimizer_1/AssignAssignnn2/bias/optimizer_1&nn2/bias/optimizer_1/Initializer/zeros*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
}
nn2/bias/optimizer_1/readIdentitynn2/bias/optimizer_1*
_output_shapes
:*
T0*
_class
loc:@nn2/bias
a
train/minimize/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
Y
train/minimize/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
train/minimize/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
train/minimize/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
Ё
:train/minimize/update_rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilontrain/gradients/AddN_10* 
_output_shapes
:
ВА*
use_locking( *
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
use_nesterov( 
Ќ
8train/minimize/update_rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamrnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilontrain/gradients/AddN_9*
_output_shapes	
:А*
use_locking( *
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
use_nesterov( 
≤
*train/minimize/update_nn1/kernel/ApplyAdam	ApplyAdam
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilon=train/gradients/nn/nn1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@nn1/kernel*
use_nesterov( *
_output_shapes
:	А 
§
(train/minimize/update_nn1/bias/ApplyAdam	ApplyAdamnn1/biasnn1/bias/optimizernn1/bias/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilon>train/gradients/nn/nn1/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@nn1/bias*
use_nesterov( 
±
*train/minimize/update_nn2/kernel/ApplyAdam	ApplyAdam
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilon=train/gradients/nn/nn2/MatMul_grad/tuple/control_dependency_1*
_class
loc:@nn2/kernel*
use_nesterov( *
_output_shapes

: *
use_locking( *
T0
§
(train/minimize/update_nn2/bias/ApplyAdam	ApplyAdamnn2/biasnn2/bias/optimizernn2/bias/optimizer_1train/beta1_power/readtrain/beta2_power/readtrain/minimize/learning_ratetrain/minimize/beta1train/minimize/beta2train/minimize/epsilon>train/gradients/nn/nn2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@nn2/bias*
use_nesterov( *
_output_shapes
:
≠
train/minimize/mulMultrain/beta1_power/readtrain/minimize/beta1)^train/minimize/update_nn1/bias/ApplyAdam+^train/minimize/update_nn1/kernel/ApplyAdam)^train/minimize/update_nn2/bias/ApplyAdam+^train/minimize/update_nn2/kernel/ApplyAdam9^train/minimize/update_rnn/basic_lstm_cell/bias/ApplyAdam;^train/minimize/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
_class
loc:@nn1/bias*
_output_shapes
: *
T0
≠
train/minimize/AssignAssigntrain/beta1_powertrain/minimize/mul*
use_locking( *
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ѓ
train/minimize/mul_1Multrain/beta2_power/readtrain/minimize/beta2)^train/minimize/update_nn1/bias/ApplyAdam+^train/minimize/update_nn1/kernel/ApplyAdam)^train/minimize/update_nn2/bias/ApplyAdam+^train/minimize/update_nn2/kernel/ApplyAdam9^train/minimize/update_rnn/basic_lstm_cell/bias/ApplyAdam;^train/minimize/update_rnn/basic_lstm_cell/kernel/ApplyAdam*
T0*
_class
loc:@nn1/bias*
_output_shapes
: 
±
train/minimize/Assign_1Assigntrain/beta2_powertrain/minimize/mul_1*
use_locking( *
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
р
train/minimizeNoOp^train/minimize/Assign^train/minimize/Assign_1)^train/minimize/update_nn1/bias/ApplyAdam+^train/minimize/update_nn1/kernel/ApplyAdam)^train/minimize/update_nn2/bias/ApplyAdam+^train/minimize/update_nn2/kernel/ApplyAdam9^train/minimize/update_rnn/basic_lstm_cell/bias/ApplyAdam;^train/minimize/update_rnn/basic_lstm_cell/kernel/ApplyAdam
ь
initNoOp^nn1/bias/Assign^nn1/bias/optimizer/Assign^nn1/bias/optimizer_1/Assign^nn1/kernel/Assign^nn1/kernel/optimizer/Assign^nn1/kernel/optimizer_1/Assign^nn2/bias/Assign^nn2/bias/optimizer/Assign^nn2/bias/optimizer_1/Assign^nn2/kernel/Assign^nn2/kernel/optimizer/Assign^nn2/kernel/optimizer_1/Assign ^rnn/basic_lstm_cell/bias/Assign*^rnn/basic_lstm_cell/bias/optimizer/Assign,^rnn/basic_lstm_cell/bias/optimizer_1/Assign"^rnn/basic_lstm_cell/kernel/Assign,^rnn/basic_lstm_cell/kernel/optimizer/Assign.^rnn/basic_lstm_cell/kernel/optimizer_1/Assign^train/beta1_power/Assign^train/beta2_power/Assign
–
Merge/MergeSummaryMergeSummaryrnn/outputs
rnn/statesrnn/rnn_weightsrnn/rnn_biasesnn/nn1_weightsnn/nn1_biasesnn/nn2_weightsnn/nn2_biasesnn/train_prediction*
N	*
_output_shapes
: 
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_c3217c1c062a41d1af8f5b67eaac61b9/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
≤
save/SaveV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
Л
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
„
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
µ
save/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
п
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ъ
save/AssignAssignnn1/biassave/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
®
save/Assign_1Assignnn1/bias/optimizersave/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
™
save/Assign_2Assignnn1/bias/optimizer_1save/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
І
save/Assign_3Assign
nn1/kernelsave/RestoreV2:3*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
±
save/Assign_4Assignnn1/kernel/optimizersave/RestoreV2:4*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
≥
save/Assign_5Assignnn1/kernel/optimizer_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ю
save/Assign_6Assignnn2/biassave/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
®
save/Assign_7Assignnn2/bias/optimizersave/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save/Assign_8Assignnn2/bias/optimizer_1save/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
¶
save/Assign_9Assign
nn2/kernelsave/RestoreV2:9*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel
≤
save/Assign_10Assignnn2/kernel/optimizersave/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
і
save/Assign_11Assignnn2/kernel/optimizer_1save/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ѕ
save/Assign_12Assignrnn/basic_lstm_cell/biassave/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
Ћ
save/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
Ќ
save/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save/RestoreV2:14*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias
 
save/Assign_15Assignrnn/basic_lstm_cell/kernelsave/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
‘
save/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
÷
save/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
•
save/Assign_18Assigntrain/beta1_powersave/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
•
save/Assign_19Assigntrain/beta2_powersave/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
в
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_4a555d488574434bb0be0140806cd5a1/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
і
save_1/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
я
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
Ј
save_1/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power
Р
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
ч
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ю
save_1/AssignAssignnn1/biassave_1/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_1/Assign_1Assignnn1/bias/optimizersave_1/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ѓ
save_1/Assign_2Assignnn1/bias/optimizer_1save_1/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_1/Assign_3Assign
nn1/kernelsave_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
µ
save_1/Assign_4Assignnn1/kernel/optimizersave_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_1/Assign_5Assignnn1/kernel/optimizer_1save_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ґ
save_1/Assign_6Assignnn2/biassave_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_1/Assign_7Assignnn2/bias/optimizersave_1/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_1/Assign_8Assignnn2/bias/optimizer_1save_1/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save_1/Assign_9Assign
nn2/kernelsave_1/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
ґ
save_1/Assign_10Assignnn2/kernel/optimizersave_1/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_1/Assign_11Assignnn2/kernel/optimizer_1save_1/RestoreV2:11*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel
≈
save_1/Assign_12Assignrnn/basic_lstm_cell/biassave_1/RestoreV2:12*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(
ѕ
save_1/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_1/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_1/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_1/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_1/Assign_15Assignrnn/basic_lstm_cell/kernelsave_1/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ў
save_1/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_1/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_1/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_1/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
©
save_1/Assign_18Assigntrain/beta1_powersave_1/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_1/Assign_19Assigntrain/beta2_powersave_1/RestoreV2:19*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
М
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_2/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_124f69ef4fc0401f8b4597068e1c00f3/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
і
save_2/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_2/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
я
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
Ј
save_2/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_2/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_2/AssignAssignnn1/biassave_2/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
ђ
save_2/Assign_1Assignnn1/bias/optimizersave_2/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ѓ
save_2/Assign_2Assignnn1/bias/optimizer_1save_2/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ђ
save_2/Assign_3Assign
nn1/kernelsave_2/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
µ
save_2/Assign_4Assignnn1/kernel/optimizersave_2/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_2/Assign_5Assignnn1/kernel/optimizer_1save_2/RestoreV2:5*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ґ
save_2/Assign_6Assignnn2/biassave_2/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_2/Assign_7Assignnn2/bias/optimizersave_2/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
Ѓ
save_2/Assign_8Assignnn2/bias/optimizer_1save_2/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
™
save_2/Assign_9Assign
nn2/kernelsave_2/RestoreV2:9*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
ґ
save_2/Assign_10Assignnn2/kernel/optimizersave_2/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_2/Assign_11Assignnn2/kernel/optimizer_1save_2/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
≈
save_2/Assign_12Assignrnn/basic_lstm_cell/biassave_2/RestoreV2:12*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
ѕ
save_2/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_2/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_2/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_2/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_2/Assign_15Assignrnn/basic_lstm_cell/kernelsave_2/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ў
save_2/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_2/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_2/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_2/RestoreV2:17*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
©
save_2/Assign_18Assigntrain/beta1_powersave_2/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_2/Assign_19Assigntrain/beta2_powersave_2/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
М
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_9bcb0e24504b48d8a75f616198de902c/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
і
save_3/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_3/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
_output_shapes
:*
T0*

axis *
N
Г
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
В
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
Ј
save_3/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
ч
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_3/AssignAssignnn1/biassave_3/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_3/Assign_1Assignnn1/bias/optimizersave_3/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ѓ
save_3/Assign_2Assignnn1/bias/optimizer_1save_3/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_3/Assign_3Assign
nn1/kernelsave_3/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
µ
save_3/Assign_4Assignnn1/kernel/optimizersave_3/RestoreV2:4*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
Ј
save_3/Assign_5Assignnn1/kernel/optimizer_1save_3/RestoreV2:5*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
Ґ
save_3/Assign_6Assignnn2/biassave_3/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_3/Assign_7Assignnn2/bias/optimizersave_3/RestoreV2:7*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ѓ
save_3/Assign_8Assignnn2/bias/optimizer_1save_3/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save_3/Assign_9Assign
nn2/kernelsave_3/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
ґ
save_3/Assign_10Assignnn2/kernel/optimizersave_3/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_3/Assign_11Assignnn2/kernel/optimizer_1save_3/RestoreV2:11*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
≈
save_3/Assign_12Assignrnn/basic_lstm_cell/biassave_3/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ѕ
save_3/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_3/RestoreV2:13*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_3/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_3/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_3/Assign_15Assignrnn/basic_lstm_cell/kernelsave_3/RestoreV2:15*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Ў
save_3/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_3/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_3/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_3/RestoreV2:17*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
©
save_3/Assign_18Assigntrain/beta1_powersave_3/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_3/Assign_19Assigntrain/beta2_powersave_3/RestoreV2:19*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
М
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_3^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_f1faf100816e49b29eae31a2ccf11ad2/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
і
save_4/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_4/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
£
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*

axis *
N*
_output_shapes
:*
T0
Г
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
В
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
Ј
save_4/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_4/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_4/AssignAssignnn1/biassave_4/RestoreV2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
ђ
save_4/Assign_1Assignnn1/bias/optimizersave_4/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_4/Assign_2Assignnn1/bias/optimizer_1save_4/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_4/Assign_3Assign
nn1/kernelsave_4/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
µ
save_4/Assign_4Assignnn1/kernel/optimizersave_4/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_4/Assign_5Assignnn1/kernel/optimizer_1save_4/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ґ
save_4/Assign_6Assignnn2/biassave_4/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_4/Assign_7Assignnn2/bias/optimizersave_4/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_4/Assign_8Assignnn2/bias/optimizer_1save_4/RestoreV2:8*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
™
save_4/Assign_9Assign
nn2/kernelsave_4/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
ґ
save_4/Assign_10Assignnn2/kernel/optimizersave_4/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_4/Assign_11Assignnn2/kernel/optimizer_1save_4/RestoreV2:11*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
≈
save_4/Assign_12Assignrnn/basic_lstm_cell/biassave_4/RestoreV2:12*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
ѕ
save_4/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_4/RestoreV2:13*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_4/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_4/RestoreV2:14*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
ќ
save_4/Assign_15Assignrnn/basic_lstm_cell/kernelsave_4/RestoreV2:15* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
Ў
save_4/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_4/RestoreV2:16*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Џ
save_4/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_4/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
©
save_4/Assign_18Assigntrain/beta1_powersave_4/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_4/Assign_19Assigntrain/beta2_powersave_4/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
М
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_3^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
shape: *
dtype0
Ж
save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_7e322e35951c493c922e8ea3dfb2cdee/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Е
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
і
save_5/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_5/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B 
я
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_5/ShardedFilename
£
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
В
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
Ј
save_5/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_5/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ю
save_5/AssignAssignnn1/biassave_5/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_5/Assign_1Assignnn1/bias/optimizersave_5/RestoreV2:1*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ѓ
save_5/Assign_2Assignnn1/bias/optimizer_1save_5/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_5/Assign_3Assign
nn1/kernelsave_5/RestoreV2:3*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel
µ
save_5/Assign_4Assignnn1/kernel/optimizersave_5/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_5/Assign_5Assignnn1/kernel/optimizer_1save_5/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ґ
save_5/Assign_6Assignnn2/biassave_5/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_5/Assign_7Assignnn2/bias/optimizersave_5/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_5/Assign_8Assignnn2/bias/optimizer_1save_5/RestoreV2:8*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
™
save_5/Assign_9Assign
nn2/kernelsave_5/RestoreV2:9*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
ґ
save_5/Assign_10Assignnn2/kernel/optimizersave_5/RestoreV2:10*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
Є
save_5/Assign_11Assignnn2/kernel/optimizer_1save_5/RestoreV2:11*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
≈
save_5/Assign_12Assignrnn/basic_lstm_cell/biassave_5/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ѕ
save_5/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_5/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_5/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_5/RestoreV2:14*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
ќ
save_5/Assign_15Assignrnn/basic_lstm_cell/kernelsave_5/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ў
save_5/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_5/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_5/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_5/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
©
save_5/Assign_18Assigntrain/beta1_powersave_5/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_5/Assign_19Assigntrain/beta2_powersave_5/RestoreV2:19*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
М
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_3^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
_output_shapes
: *
shape: 
Ж
save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_aa5a44bc5c3a431eb0c3f92cd3b03a07/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
і
save_6/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_6/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
£
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
В
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
Ј
save_6/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_6/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
Ю
save_6/AssignAssignnn1/biassave_6/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_6/Assign_1Assignnn1/bias/optimizersave_6/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_6/Assign_2Assignnn1/bias/optimizer_1save_6/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_6/Assign_3Assign
nn1/kernelsave_6/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
µ
save_6/Assign_4Assignnn1/kernel/optimizersave_6/RestoreV2:4*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_6/Assign_5Assignnn1/kernel/optimizer_1save_6/RestoreV2:5*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel
Ґ
save_6/Assign_6Assignnn2/biassave_6/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_6/Assign_7Assignnn2/bias/optimizersave_6/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_6/Assign_8Assignnn2/bias/optimizer_1save_6/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
™
save_6/Assign_9Assign
nn2/kernelsave_6/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
ґ
save_6/Assign_10Assignnn2/kernel/optimizersave_6/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_6/Assign_11Assignnn2/kernel/optimizer_1save_6/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
≈
save_6/Assign_12Assignrnn/basic_lstm_cell/biassave_6/RestoreV2:12*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias
ѕ
save_6/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_6/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_6/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_6/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_6/Assign_15Assignrnn/basic_lstm_cell/kernelsave_6/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ў
save_6/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_6/RestoreV2:16*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Џ
save_6/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_6/RestoreV2:17*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
©
save_6/Assign_18Assigntrain/beta1_powersave_6/RestoreV2:18*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias
©
save_6/Assign_19Assigntrain/beta2_powersave_6/RestoreV2:19*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
М
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_3^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_7/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_cc2cd5f0e92d4a32b20b2d298012e140/part*
dtype0
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
і
save_7/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_7/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
£
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
N*
_output_shapes
:*
T0
Г
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
В
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
_output_shapes
: *
T0
Ј
save_7/RestoreV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
Р
!save_7/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_7/AssignAssignnn1/biassave_7/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_7/Assign_1Assignnn1/bias/optimizersave_7/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ѓ
save_7/Assign_2Assignnn1/bias/optimizer_1save_7/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_7/Assign_3Assign
nn1/kernelsave_7/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
µ
save_7/Assign_4Assignnn1/kernel/optimizersave_7/RestoreV2:4*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
Ј
save_7/Assign_5Assignnn1/kernel/optimizer_1save_7/RestoreV2:5*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
Ґ
save_7/Assign_6Assignnn2/biassave_7/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_7/Assign_7Assignnn2/bias/optimizersave_7/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
Ѓ
save_7/Assign_8Assignnn2/bias/optimizer_1save_7/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save_7/Assign_9Assign
nn2/kernelsave_7/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
ґ
save_7/Assign_10Assignnn2/kernel/optimizersave_7/RestoreV2:10*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
Є
save_7/Assign_11Assignnn2/kernel/optimizer_1save_7/RestoreV2:11*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
≈
save_7/Assign_12Assignrnn/basic_lstm_cell/biassave_7/RestoreV2:12*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
ѕ
save_7/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_7/RestoreV2:13*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_7/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_7/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_7/Assign_15Assignrnn/basic_lstm_cell/kernelsave_7/RestoreV2:15* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
Ў
save_7/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_7/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_7/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_7/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
©
save_7/Assign_18Assigntrain/beta1_powersave_7/RestoreV2:18*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
©
save_7/Assign_19Assigntrain/beta2_powersave_7/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
М
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_3^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
dtype0*
_output_shapes
: *
shape: 
Ж
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_a0973e123e5846f8987f9c5595a0e640/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_8/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
і
save_8/SaveV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
Н
save_8/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*)
_class
loc:@save_8/ShardedFilename*
_output_shapes
: *
T0
£
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*

axis *
N*
_output_shapes
:
Г
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
В
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
_output_shapes
: *
T0
Ј
save_8/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_8/RestoreV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
ч
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_8/AssignAssignnn1/biassave_8/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
ђ
save_8/Assign_1Assignnn1/bias/optimizersave_8/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_8/Assign_2Assignnn1/bias/optimizer_1save_8/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_8/Assign_3Assign
nn1/kernelsave_8/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
µ
save_8/Assign_4Assignnn1/kernel/optimizersave_8/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_8/Assign_5Assignnn1/kernel/optimizer_1save_8/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ґ
save_8/Assign_6Assignnn2/biassave_8/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_8/Assign_7Assignnn2/bias/optimizersave_8/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
Ѓ
save_8/Assign_8Assignnn2/bias/optimizer_1save_8/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save_8/Assign_9Assign
nn2/kernelsave_8/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
ґ
save_8/Assign_10Assignnn2/kernel/optimizersave_8/RestoreV2:10*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
Є
save_8/Assign_11Assignnn2/kernel/optimizer_1save_8/RestoreV2:11*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
≈
save_8/Assign_12Assignrnn/basic_lstm_cell/biassave_8/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ѕ
save_8/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_8/RestoreV2:13*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_8/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_8/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_8/Assign_15Assignrnn/basic_lstm_cell/kernelsave_8/RestoreV2:15*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Ў
save_8/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_8/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_8/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_8/RestoreV2:17* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
©
save_8/Assign_18Assigntrain/beta1_powersave_8/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
©
save_8/Assign_19Assigntrain/beta2_powersave_8/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
М
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_3^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
shape: *
dtype0*
_output_shapes
: 
Ж
save_9/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_fc6ea646297f4d7082e6c731e732f565/part*
dtype0
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_9/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
і
save_9/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Н
save_9/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
я
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Щ
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_9/ShardedFilename
£
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
_output_shapes
:*
T0*

axis *
N
Г
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
В
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
Ј
save_9/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
Р
!save_9/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ч
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
Ю
save_9/AssignAssignnn1/biassave_9/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
ђ
save_9/Assign_1Assignnn1/bias/optimizersave_9/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_9/Assign_2Assignnn1/bias/optimizer_1save_9/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ђ
save_9/Assign_3Assign
nn1/kernelsave_9/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
µ
save_9/Assign_4Assignnn1/kernel/optimizersave_9/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_9/Assign_5Assignnn1/kernel/optimizer_1save_9/RestoreV2:5*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ґ
save_9/Assign_6Assignnn2/biassave_9/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_9/Assign_7Assignnn2/bias/optimizersave_9/RestoreV2:7*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ѓ
save_9/Assign_8Assignnn2/bias/optimizer_1save_9/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
™
save_9/Assign_9Assign
nn2/kernelsave_9/RestoreV2:9*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
ґ
save_9/Assign_10Assignnn2/kernel/optimizersave_9/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_9/Assign_11Assignnn2/kernel/optimizer_1save_9/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
≈
save_9/Assign_12Assignrnn/basic_lstm_cell/biassave_9/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ѕ
save_9/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_9/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_9/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_9/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
ќ
save_9/Assign_15Assignrnn/basic_lstm_cell/kernelsave_9/RestoreV2:15* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
Ў
save_9/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_9/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_9/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_9/RestoreV2:17*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
©
save_9/Assign_18Assigntrain/beta1_powersave_9/RestoreV2:18*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
©
save_9/Assign_19Assigntrain/beta2_powersave_9/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
М
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_3^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
shape: *
dtype0*
_output_shapes
: 
З
save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_baa7a271e12f4b208ca336799c10fa1f/part*
dtype0*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
µ
save_10/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_10/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: 
¶
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*

axis *
N*
_output_shapes
:*
T0
Ж
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
Ж
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
Є
save_10/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_10/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
†
save_10/AssignAssignnn1/biassave_10/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ѓ
save_10/Assign_1Assignnn1/bias/optimizersave_10/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
∞
save_10/Assign_2Assignnn1/bias/optimizer_1save_10/RestoreV2:2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
≠
save_10/Assign_3Assign
nn1/kernelsave_10/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_10/Assign_4Assignnn1/kernel/optimizersave_10/RestoreV2:4*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
є
save_10/Assign_5Assignnn1/kernel/optimizer_1save_10/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
§
save_10/Assign_6Assignnn2/biassave_10/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
Ѓ
save_10/Assign_7Assignnn2/bias/optimizersave_10/RestoreV2:7*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
∞
save_10/Assign_8Assignnn2/bias/optimizer_1save_10/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_10/Assign_9Assign
nn2/kernelsave_10/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_10/Assign_10Assignnn2/kernel/optimizersave_10/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ї
save_10/Assign_11Assignnn2/kernel/optimizer_1save_10/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
«
save_10/Assign_12Assignrnn/basic_lstm_cell/biassave_10/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_10/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_10/RestoreV2:13*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
”
save_10/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_10/RestoreV2:14*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
–
save_10/Assign_15Assignrnn/basic_lstm_cell/kernelsave_10/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_10/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_10/RestoreV2:16*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
№
save_10/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_10/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ђ
save_10/Assign_18Assigntrain/beta1_powersave_10/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_10/Assign_19Assigntrain/beta2_powersave_10/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
°
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_3^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_11/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_7dcd426feb944e97af32e052e91adf78/part*
dtype0
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_11/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
µ
save_11/SaveV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
О
save_11/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: *
T0
¶
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*

axis *
N*
_output_shapes
:*
T0
Ж
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
Ж
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
Є
save_11/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_11/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_11/AssignAssignnn1/biassave_11/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ѓ
save_11/Assign_1Assignnn1/bias/optimizersave_11/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
∞
save_11/Assign_2Assignnn1/bias/optimizer_1save_11/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
≠
save_11/Assign_3Assign
nn1/kernelsave_11/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_11/Assign_4Assignnn1/kernel/optimizersave_11/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_11/Assign_5Assignnn1/kernel/optimizer_1save_11/RestoreV2:5*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
§
save_11/Assign_6Assignnn2/biassave_11/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_11/Assign_7Assignnn2/bias/optimizersave_11/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
∞
save_11/Assign_8Assignnn2/bias/optimizer_1save_11/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_11/Assign_9Assign
nn2/kernelsave_11/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
Є
save_11/Assign_10Assignnn2/kernel/optimizersave_11/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ї
save_11/Assign_11Assignnn2/kernel/optimizer_1save_11/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
«
save_11/Assign_12Assignrnn/basic_lstm_cell/biassave_11/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_11/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_11/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
”
save_11/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_11/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
–
save_11/Assign_15Assignrnn/basic_lstm_cell/kernelsave_11/RestoreV2:15*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Џ
save_11/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_11/RestoreV2:16* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
№
save_11/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_11/RestoreV2:17* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
Ђ
save_11/Assign_18Assigntrain/beta1_powersave_11/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_11/Assign_19Assigntrain/beta2_powersave_11/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
°
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_3^save_11/Assign_4^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
_output_shapes
: *
shape: *
dtype0
З
save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_708f9e5cbf9444b399e9f156c9a9ba89/part*
dtype0*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Й
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
µ
save_12/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_12/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0**
_class 
loc:@save_12/ShardedFilename*
_output_shapes
: 
¶
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*

axis *
N*
_output_shapes
:*
T0
Ж
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
Ж
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
_output_shapes
: *
T0
Є
save_12/RestoreV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
С
"save_12/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
†
save_12/AssignAssignnn1/biassave_12/RestoreV2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_12/Assign_1Assignnn1/bias/optimizersave_12/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
∞
save_12/Assign_2Assignnn1/bias/optimizer_1save_12/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
≠
save_12/Assign_3Assign
nn1/kernelsave_12/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_12/Assign_4Assignnn1/kernel/optimizersave_12/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_12/Assign_5Assignnn1/kernel/optimizer_1save_12/RestoreV2:5*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
§
save_12/Assign_6Assignnn2/biassave_12/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_12/Assign_7Assignnn2/bias/optimizersave_12/RestoreV2:7*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
∞
save_12/Assign_8Assignnn2/bias/optimizer_1save_12/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_12/Assign_9Assign
nn2/kernelsave_12/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_12/Assign_10Assignnn2/kernel/optimizersave_12/RestoreV2:10*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
Ї
save_12/Assign_11Assignnn2/kernel/optimizer_1save_12/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
«
save_12/Assign_12Assignrnn/basic_lstm_cell/biassave_12/RestoreV2:12*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
—
save_12/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_12/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
”
save_12/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_12/RestoreV2:14*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
–
save_12/Assign_15Assignrnn/basic_lstm_cell/kernelsave_12/RestoreV2:15*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Џ
save_12/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_12/RestoreV2:16*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
№
save_12/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_12/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ђ
save_12/Assign_18Assigntrain/beta1_powersave_12/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_12/Assign_19Assigntrain/beta2_powersave_12/RestoreV2:19*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
°
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_3^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_1d6f7ff453c741f19ac05beb551c942e/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_13/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
µ
save_13/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_13/SaveV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
г
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: 
¶
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
_output_shapes
:*
T0*

axis *
N
Ж
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
Ж
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
Є
save_13/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_13/RestoreV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
ы
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_13/AssignAssignnn1/biassave_13/RestoreV2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_13/Assign_1Assignnn1/bias/optimizersave_13/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
∞
save_13/Assign_2Assignnn1/bias/optimizer_1save_13/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
≠
save_13/Assign_3Assign
nn1/kernelsave_13/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_13/Assign_4Assignnn1/kernel/optimizersave_13/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_13/Assign_5Assignnn1/kernel/optimizer_1save_13/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
§
save_13/Assign_6Assignnn2/biassave_13/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_13/Assign_7Assignnn2/bias/optimizersave_13/RestoreV2:7*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
∞
save_13/Assign_8Assignnn2/bias/optimizer_1save_13/RestoreV2:8*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
ђ
save_13/Assign_9Assign
nn2/kernelsave_13/RestoreV2:9*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
Є
save_13/Assign_10Assignnn2/kernel/optimizersave_13/RestoreV2:10*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
Ї
save_13/Assign_11Assignnn2/kernel/optimizer_1save_13/RestoreV2:11*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
«
save_13/Assign_12Assignrnn/basic_lstm_cell/biassave_13/RestoreV2:12*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
—
save_13/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_13/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
”
save_13/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_13/RestoreV2:14*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
–
save_13/Assign_15Assignrnn/basic_lstm_cell/kernelsave_13/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_13/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_13/RestoreV2:16*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
№
save_13/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_13/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ђ
save_13/Assign_18Assigntrain/beta1_powersave_13/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_13/Assign_19Assigntrain/beta2_powersave_13/RestoreV2:19*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
°
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_3^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
shape: *
dtype0*
_output_shapes
: 
З
save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_c874726b8ab145e3bddc6a1b3829fc7a/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
µ
save_14/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_14/SaveV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
г
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: *
T0
¶
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
_output_shapes
:*
T0*

axis *
N
Ж
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
Ж
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
Є
save_14/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_14/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_14/AssignAssignnn1/biassave_14/RestoreV2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ѓ
save_14/Assign_1Assignnn1/bias/optimizersave_14/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
∞
save_14/Assign_2Assignnn1/bias/optimizer_1save_14/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
≠
save_14/Assign_3Assign
nn1/kernelsave_14/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_14/Assign_4Assignnn1/kernel/optimizersave_14/RestoreV2:4*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
є
save_14/Assign_5Assignnn1/kernel/optimizer_1save_14/RestoreV2:5*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
§
save_14/Assign_6Assignnn2/biassave_14/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_14/Assign_7Assignnn2/bias/optimizersave_14/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
∞
save_14/Assign_8Assignnn2/bias/optimizer_1save_14/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_14/Assign_9Assign
nn2/kernelsave_14/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
Є
save_14/Assign_10Assignnn2/kernel/optimizersave_14/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ї
save_14/Assign_11Assignnn2/kernel/optimizer_1save_14/RestoreV2:11*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
«
save_14/Assign_12Assignrnn/basic_lstm_cell/biassave_14/RestoreV2:12*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
—
save_14/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_14/RestoreV2:13*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
”
save_14/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_14/RestoreV2:14*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
–
save_14/Assign_15Assignrnn/basic_lstm_cell/kernelsave_14/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_14/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_14/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
№
save_14/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_14/RestoreV2:17*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Ђ
save_14/Assign_18Assigntrain/beta1_powersave_14/RestoreV2:18*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ђ
save_14/Assign_19Assigntrain/beta2_powersave_14/RestoreV2:19*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
°
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_3^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
shape: *
dtype0
З
save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_b0bc5e0e6113433f9c2a94c854dcf30c/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
µ
save_15/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_15/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_15/ShardedFilename
¶
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*

axis *
N*
_output_shapes
:*
T0
Ж
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
Ж
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
_output_shapes
: *
T0
Є
save_15/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_15/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_15/AssignAssignnn1/biassave_15/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ѓ
save_15/Assign_1Assignnn1/bias/optimizersave_15/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
∞
save_15/Assign_2Assignnn1/bias/optimizer_1save_15/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
≠
save_15/Assign_3Assign
nn1/kernelsave_15/RestoreV2:3*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
Ј
save_15/Assign_4Assignnn1/kernel/optimizersave_15/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_15/Assign_5Assignnn1/kernel/optimizer_1save_15/RestoreV2:5*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0
§
save_15/Assign_6Assignnn2/biassave_15/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_15/Assign_7Assignnn2/bias/optimizersave_15/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
∞
save_15/Assign_8Assignnn2/bias/optimizer_1save_15/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_15/Assign_9Assign
nn2/kernelsave_15/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(
Є
save_15/Assign_10Assignnn2/kernel/optimizersave_15/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ї
save_15/Assign_11Assignnn2/kernel/optimizer_1save_15/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
«
save_15/Assign_12Assignrnn/basic_lstm_cell/biassave_15/RestoreV2:12*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
—
save_15/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_15/RestoreV2:13*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(
”
save_15/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_15/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
–
save_15/Assign_15Assignrnn/basic_lstm_cell/kernelsave_15/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_15/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_15/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
№
save_15/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_15/RestoreV2:17*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Ђ
save_15/Assign_18Assigntrain/beta1_powersave_15/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_15/Assign_19Assigntrain/beta2_powersave_15/RestoreV2:19*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
°
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_3^save_15/Assign_4^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard
\
save_16/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_16/filenamePlaceholderWithDefaultsave_16/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_16/ConstPlaceholderWithDefaultsave_16/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_0ed4f289080944ada9d6ffe83e4ebb9a/part*
dtype0*
_output_shapes
: 
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
µ
save_16/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_16/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_16/ShardedFilename
¶
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ж
save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(
Ж
save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
_output_shapes
: *
T0
Є
save_16/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_16/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*"
dtypes
2*d
_output_shapesR
P::::::::::::::::::::
†
save_16/AssignAssignnn1/biassave_16/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
Ѓ
save_16/Assign_1Assignnn1/bias/optimizersave_16/RestoreV2:1*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
∞
save_16/Assign_2Assignnn1/bias/optimizer_1save_16/RestoreV2:2*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
≠
save_16/Assign_3Assign
nn1/kernelsave_16/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_16/Assign_4Assignnn1/kernel/optimizersave_16/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_16/Assign_5Assignnn1/kernel/optimizer_1save_16/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
§
save_16/Assign_6Assignnn2/biassave_16/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_16/Assign_7Assignnn2/bias/optimizersave_16/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
∞
save_16/Assign_8Assignnn2/bias/optimizer_1save_16/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_16/Assign_9Assign
nn2/kernelsave_16/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_16/Assign_10Assignnn2/kernel/optimizersave_16/RestoreV2:10*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
Ї
save_16/Assign_11Assignnn2/kernel/optimizer_1save_16/RestoreV2:11*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
«
save_16/Assign_12Assignrnn/basic_lstm_cell/biassave_16/RestoreV2:12*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_16/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_16/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
”
save_16/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_16/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
–
save_16/Assign_15Assignrnn/basic_lstm_cell/kernelsave_16/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_16/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_16/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
№
save_16/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_16/RestoreV2:17* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(
Ђ
save_16/Assign_18Assigntrain/beta1_powersave_16/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_16/Assign_19Assigntrain/beta2_powersave_16/RestoreV2:19*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
°
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_16^save_16/Assign_17^save_16/Assign_18^save_16/Assign_19^save_16/Assign_2^save_16/Assign_3^save_16/Assign_4^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard
\
save_17/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_17/filenamePlaceholderWithDefaultsave_17/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_17/ConstPlaceholderWithDefaultsave_17/filename*
shape: *
dtype0*
_output_shapes
: 
З
save_17/StringJoin/inputs_1Const*<
value3B1 B+_temp_2a2a40ae610d45ac91ef83b6ca8fa608/part*
dtype0*
_output_shapes
: 
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
µ
save_17/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_17/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_17/ShardedFilename
¶
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ж
save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(
Ж
save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
Є
save_17/RestoreV2/tensor_namesConst*
_output_shapes
:*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0
С
"save_17/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ы
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_17/AssignAssignnn1/biassave_17/RestoreV2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ѓ
save_17/Assign_1Assignnn1/bias/optimizersave_17/RestoreV2:1*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
∞
save_17/Assign_2Assignnn1/bias/optimizer_1save_17/RestoreV2:2*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
≠
save_17/Assign_3Assign
nn1/kernelsave_17/RestoreV2:3*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
Ј
save_17/Assign_4Assignnn1/kernel/optimizersave_17/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_17/Assign_5Assignnn1/kernel/optimizer_1save_17/RestoreV2:5*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(
§
save_17/Assign_6Assignnn2/biassave_17/RestoreV2:6*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Ѓ
save_17/Assign_7Assignnn2/bias/optimizersave_17/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
∞
save_17/Assign_8Assignnn2/bias/optimizer_1save_17/RestoreV2:8*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(
ђ
save_17/Assign_9Assign
nn2/kernelsave_17/RestoreV2:9*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Є
save_17/Assign_10Assignnn2/kernel/optimizersave_17/RestoreV2:10*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
Ї
save_17/Assign_11Assignnn2/kernel/optimizer_1save_17/RestoreV2:11*
use_locking(*
T0*
_class
loc:@nn2/kernel*
validate_shape(*
_output_shapes

: 
«
save_17/Assign_12Assignrnn/basic_lstm_cell/biassave_17/RestoreV2:12*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0
—
save_17/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_17/RestoreV2:13*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
”
save_17/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_17/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias*
validate_shape(*
_output_shapes	
:А
–
save_17/Assign_15Assignrnn/basic_lstm_cell/kernelsave_17/RestoreV2:15*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
Џ
save_17/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_17/RestoreV2:16*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА
№
save_17/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_17/RestoreV2:17*-
_class#
!loc:@rnn/basic_lstm_cell/kernel*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0
Ђ
save_17/Assign_18Assigntrain/beta1_powersave_17/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_17/Assign_19Assigntrain/beta2_powersave_17/RestoreV2:19*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(
°
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_16^save_17/Assign_17^save_17/Assign_18^save_17/Assign_19^save_17/Assign_2^save_17/Assign_3^save_17/Assign_4^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard
\
save_18/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_18/filenamePlaceholderWithDefaultsave_18/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_18/ConstPlaceholderWithDefaultsave_18/filename*
dtype0*
_output_shapes
: *
shape: 
З
save_18/StringJoin/inputs_1Const*<
value3B1 B+_temp_e6cba86526b2404a9ecbdec38cf0e068/part*
dtype0*
_output_shapes
: 
~
save_18/StringJoin
StringJoinsave_18/Constsave_18/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_18/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
_
save_18/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Й
save_18/ShardedFilenameShardedFilenamesave_18/StringJoinsave_18/ShardedFilename/shardsave_18/num_shards*
_output_shapes
: 
µ
save_18/SaveV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
О
save_18/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
г
save_18/SaveV2SaveV2save_18/ShardedFilenamesave_18/SaveV2/tensor_namessave_18/SaveV2/shape_and_slicesnn1/biasnn1/bias/optimizernn1/bias/optimizer_1
nn1/kernelnn1/kernel/optimizernn1/kernel/optimizer_1nn2/biasnn2/bias/optimizernn2/bias/optimizer_1
nn2/kernelnn2/kernel/optimizernn2/kernel/optimizer_1rnn/basic_lstm_cell/bias"rnn/basic_lstm_cell/bias/optimizer$rnn/basic_lstm_cell/bias/optimizer_1rnn/basic_lstm_cell/kernel$rnn/basic_lstm_cell/kernel/optimizer&rnn/basic_lstm_cell/kernel/optimizer_1train/beta1_powertrain/beta2_power*"
dtypes
2
Э
save_18/control_dependencyIdentitysave_18/ShardedFilename^save_18/SaveV2*
T0**
_class 
loc:@save_18/ShardedFilename*
_output_shapes
: 
¶
.save_18/MergeV2Checkpoints/checkpoint_prefixesPacksave_18/ShardedFilename^save_18/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ж
save_18/MergeV2CheckpointsMergeV2Checkpoints.save_18/MergeV2Checkpoints/checkpoint_prefixessave_18/Const*
delete_old_dirs(
Ж
save_18/IdentityIdentitysave_18/Const^save_18/MergeV2Checkpoints^save_18/control_dependency*
T0*
_output_shapes
: 
Є
save_18/RestoreV2/tensor_namesConst*е
valueџBЎBnn1/biasBnn1/bias/optimizerBnn1/bias/optimizer_1B
nn1/kernelBnn1/kernel/optimizerBnn1/kernel/optimizer_1Bnn2/biasBnn2/bias/optimizerBnn2/bias/optimizer_1B
nn2/kernelBnn2/kernel/optimizerBnn2/kernel/optimizer_1Brnn/basic_lstm_cell/biasB"rnn/basic_lstm_cell/bias/optimizerB$rnn/basic_lstm_cell/bias/optimizer_1Brnn/basic_lstm_cell/kernelB$rnn/basic_lstm_cell/kernel/optimizerB&rnn/basic_lstm_cell/kernel/optimizer_1Btrain/beta1_powerBtrain/beta2_power*
dtype0*
_output_shapes
:
С
"save_18/RestoreV2/shape_and_slicesConst*
_output_shapes
:*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
ы
save_18/RestoreV2	RestoreV2save_18/Constsave_18/RestoreV2/tensor_names"save_18/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
†
save_18/AssignAssignnn1/biassave_18/RestoreV2*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Ѓ
save_18/Assign_1Assignnn1/bias/optimizersave_18/RestoreV2:1*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: *
use_locking(
∞
save_18/Assign_2Assignnn1/bias/optimizer_1save_18/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias
≠
save_18/Assign_3Assign
nn1/kernelsave_18/RestoreV2:3*
validate_shape(*
_output_shapes
:	А *
use_locking(*
T0*
_class
loc:@nn1/kernel
Ј
save_18/Assign_4Assignnn1/kernel/optimizersave_18/RestoreV2:4*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
є
save_18/Assign_5Assignnn1/kernel/optimizer_1save_18/RestoreV2:5*
use_locking(*
T0*
_class
loc:@nn1/kernel*
validate_shape(*
_output_shapes
:	А 
§
save_18/Assign_6Assignnn2/biassave_18/RestoreV2:6*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
Ѓ
save_18/Assign_7Assignnn2/bias/optimizersave_18/RestoreV2:7*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
∞
save_18/Assign_8Assignnn2/bias/optimizer_1save_18/RestoreV2:8*
use_locking(*
T0*
_class
loc:@nn2/bias*
validate_shape(*
_output_shapes
:
ђ
save_18/Assign_9Assign
nn2/kernelsave_18/RestoreV2:9*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel
Є
save_18/Assign_10Assignnn2/kernel/optimizersave_18/RestoreV2:10*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel
Ї
save_18/Assign_11Assignnn2/kernel/optimizer_1save_18/RestoreV2:11*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*
_class
loc:@nn2/kernel
«
save_18/Assign_12Assignrnn/basic_lstm_cell/biassave_18/RestoreV2:12*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias
—
save_18/Assign_13Assign"rnn/basic_lstm_cell/bias/optimizersave_18/RestoreV2:13*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias
”
save_18/Assign_14Assign$rnn/basic_lstm_cell/bias/optimizer_1save_18/RestoreV2:14*
validate_shape(*
_output_shapes	
:А*
use_locking(*
T0*+
_class!
loc:@rnn/basic_lstm_cell/bias
–
save_18/Assign_15Assignrnn/basic_lstm_cell/kernelsave_18/RestoreV2:15*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
Џ
save_18/Assign_16Assign$rnn/basic_lstm_cell/kernel/optimizersave_18/RestoreV2:16*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
№
save_18/Assign_17Assign&rnn/basic_lstm_cell/kernel/optimizer_1save_18/RestoreV2:17*
validate_shape(* 
_output_shapes
:
ВА*
use_locking(*
T0*-
_class#
!loc:@rnn/basic_lstm_cell/kernel
Ђ
save_18/Assign_18Assigntrain/beta1_powersave_18/RestoreV2:18*
use_locking(*
T0*
_class
loc:@nn1/bias*
validate_shape(*
_output_shapes
: 
Ђ
save_18/Assign_19Assigntrain/beta2_powersave_18/RestoreV2:19*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@nn1/bias
°
save_18/restore_shardNoOp^save_18/Assign^save_18/Assign_1^save_18/Assign_10^save_18/Assign_11^save_18/Assign_12^save_18/Assign_13^save_18/Assign_14^save_18/Assign_15^save_18/Assign_16^save_18/Assign_17^save_18/Assign_18^save_18/Assign_19^save_18/Assign_2^save_18/Assign_3^save_18/Assign_4^save_18/Assign_5^save_18/Assign_6^save_18/Assign_7^save_18/Assign_8^save_18/Assign_9
3
save_18/restore_allNoOp^save_18/restore_shard "E
save_18/Const:0save_18/Identity:0save_18/restore_all (5 @F8"O
aJ
H
loss/train_loss:0
loss/train_loss:0
loss/val_loss:0
loss/val_loss:0"+
e&
$
loss/train_loss:0
loss/val_loss:0"р
	variablesвя
Я
rnn/basic_lstm_cell/kernel:0!rnn/basic_lstm_cell/kernel/Assign!rnn/basic_lstm_cell/kernel/read:027rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
О
rnn/basic_lstm_cell/bias:0rnn/basic_lstm_cell/bias/Assignrnn/basic_lstm_cell/bias/read:02,rnn/basic_lstm_cell/bias/Initializer/zeros:08
_
nn1/kernel:0nn1/kernel/Assignnn1/kernel/read:02'nn1/kernel/Initializer/random_uniform:08
N

nn1/bias:0nn1/bias/Assignnn1/bias/read:02nn1/bias/Initializer/zeros:08
_
nn2/kernel:0nn2/kernel/Assignnn2/kernel/read:02'nn2/kernel/Initializer/random_uniform:08
N

nn2/bias:0nn2/bias/Assignnn2/bias/read:02nn2/bias/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
Љ
&rnn/basic_lstm_cell/kernel/optimizer:0+rnn/basic_lstm_cell/kernel/optimizer/Assign+rnn/basic_lstm_cell/kernel/optimizer/read:028rnn/basic_lstm_cell/kernel/optimizer/Initializer/zeros:0
ƒ
(rnn/basic_lstm_cell/kernel/optimizer_1:0-rnn/basic_lstm_cell/kernel/optimizer_1/Assign-rnn/basic_lstm_cell/kernel/optimizer_1/read:02:rnn/basic_lstm_cell/kernel/optimizer_1/Initializer/zeros:0
і
$rnn/basic_lstm_cell/bias/optimizer:0)rnn/basic_lstm_cell/bias/optimizer/Assign)rnn/basic_lstm_cell/bias/optimizer/read:026rnn/basic_lstm_cell/bias/optimizer/Initializer/zeros:0
Љ
&rnn/basic_lstm_cell/bias/optimizer_1:0+rnn/basic_lstm_cell/bias/optimizer_1/Assign+rnn/basic_lstm_cell/bias/optimizer_1/read:028rnn/basic_lstm_cell/bias/optimizer_1/Initializer/zeros:0
|
nn1/kernel/optimizer:0nn1/kernel/optimizer/Assignnn1/kernel/optimizer/read:02(nn1/kernel/optimizer/Initializer/zeros:0
Д
nn1/kernel/optimizer_1:0nn1/kernel/optimizer_1/Assignnn1/kernel/optimizer_1/read:02*nn1/kernel/optimizer_1/Initializer/zeros:0
t
nn1/bias/optimizer:0nn1/bias/optimizer/Assignnn1/bias/optimizer/read:02&nn1/bias/optimizer/Initializer/zeros:0
|
nn1/bias/optimizer_1:0nn1/bias/optimizer_1/Assignnn1/bias/optimizer_1/read:02(nn1/bias/optimizer_1/Initializer/zeros:0
|
nn2/kernel/optimizer:0nn2/kernel/optimizer/Assignnn2/kernel/optimizer/read:02(nn2/kernel/optimizer/Initializer/zeros:0
Д
nn2/kernel/optimizer_1:0nn2/kernel/optimizer_1/Assignnn2/kernel/optimizer_1/read:02*nn2/kernel/optimizer_1/Initializer/zeros:0
t
nn2/bias/optimizer:0nn2/bias/optimizer/Assignnn2/bias/optimizer/read:02&nn2/bias/optimizer/Initializer/zeros:0
|
nn2/bias/optimizer_1:0nn2/bias/optimizer_1/Assignnn2/bias/optimizer_1/read:02(nn2/bias/optimizer_1/Initializer/zeros:0"+
i&
$
loss/train_loss:0
loss/val_loss:0"
train_op

train/minimize"+
l&
$
loss/train_loss:0
loss/val_loss:0"+
n&
$
loss/train_loss:0
loss/val_loss:0"+
o&
$
loss/train_loss:0
loss/val_loss:0"∞
	summariesҐ
Я
rnn/outputs:0
rnn/states:0
rnn/rnn_weights:0
rnn/rnn_biases:0
nn/nn1_weights:0
nn/nn1_biases:0
nn/nn2_weights:0
nn/nn2_biases:0
nn/train_prediction:0"∞
trainable_variablesШХ
Я
rnn/basic_lstm_cell/kernel:0!rnn/basic_lstm_cell/kernel/Assign!rnn/basic_lstm_cell/kernel/read:027rnn/basic_lstm_cell/kernel/Initializer/random_uniform:08
О
rnn/basic_lstm_cell/bias:0rnn/basic_lstm_cell/bias/Assignrnn/basic_lstm_cell/bias/read:02,rnn/basic_lstm_cell/bias/Initializer/zeros:08
_
nn1/kernel:0nn1/kernel/Assignnn1/kernel/read:02'nn1/kernel/Initializer/random_uniform:08
N

nn1/bias:0nn1/bias/Assignnn1/bias/read:02nn1/bias/Initializer/zeros:08
_
nn2/kernel:0nn2/kernel/Assignnn2/kernel/read:02'nn2/kernel/Initializer/random_uniform:08
N

nn2/bias:0nn2/bias/Assignnn2/bias/read:02nn2/bias/Initializer/zeros:08"+
t&
$
loss/train_loss:0
loss/val_loss:0"+
u&
$
loss/train_loss:0
loss/val_loss:0"+
v&
$
loss/train_loss:0
loss/val_loss:0*u
serving_defaultb
#
x
X:0€€€€€€€€€

y
Y:0€€€€€€€€€tensorflow/serving/predict