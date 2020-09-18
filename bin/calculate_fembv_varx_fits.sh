#!/bin/bash

# Fit FEM-BV-VARX model to PCs of NCEPv1 500 hPa geopotential height anomalies

BASEDIR=$(dirname $0)
ABSBASEDIR=$(cd "$BASEDIR"; pwd -P)

PROJECT_DIR="${ABSBASEDIR}/.."
BIN_DIR="${PROJECT_DIR}/bin"
RESULTS_DIR="${PROJECT_DIR}/results"
EOFS_RESULTS_DIR="${RESULTS_DIR}/eofs"
EOFS_NC_DIR="${EOFS_RESULTS_DIR}/nc"
FEMBV_RESULTS_DIR="${RESULTS_DIR}/fembv_varx_fits"

if test ! -d "$RESULTS_DIR" ; then
   mkdir "$RESULTS_DIR"
fi

if test ! -d "$EOFS_RESULTS_DIR" ; then
   mkdir "$EOFS_RESULTS_DIR"
fi

if test ! -d "$EOFS_NC_DIR" ; then
   mkdir "$EOFS_NC_DIR"
fi

if test ! -d "$FEMBV_RESULTS_DIR" ; then
   mkdir "$FEMBV_RESULTS_DIR"
fi

PYTHON="python"
FIT_MODEL="${BIN_DIR}/fit_nnr1_h500_pcs_fembv_varx.py"

VAR_NAME="hgt.500"
TIMESPAN="1948_2018"
CLIMATOLOGY_BASE_PERIOD="19790101_20181231"
EOFS_BASE_PERIOD="${CLIMATOLOGY_BASE_PERIOD}"
MAX_EOFS="200"
LAT_WEIGHTS="scos"
EOFS_NORMALIZATION="unit"

hemisphere="NH"
region="atlantic"
season="ALL"

n_features="20"
n_components="3"
order="1"
state_length=""
presample_length=""

tol="1e-3"
max_iter="1000"
reg_covar="1e-6"
n_init="20"

fit_period_start="1979-01-01"
fit_period_end="2018-12-31"

random_seed="0"
verbose="yes"
loss="least_squares"
pickle="no"
standardize="no"
standardize_by=""
cross_validate="yes"
n_folds="10"

if test $# -gt 0 ; then

   while test ! "x$1" = "x" ; do
      case "$1" in
         -*=*) optarg=$(echo "$1" | sed 's/[-_a-zA-Z0-0]*=//') ;;
         *) optarg= ;;
      esac

      case $1 in
         --hemisphere=*)       hemisphere=$optarg ;;
         --region=*)           region=$optarg ;;
         --season=*)           season=$optarg ;;
         --n-features=*)       n_features=$optarg ;;
         --n-components=*)     n_components=$optarg ;;
         --order=*)            order=$optarg ;;
         --state-length=*)     state_length=$optarg ;;
         --presample-length=*) presample_length=$optarg ;;
         --tol=*)              tol=$optarg ;;
         --max-iter=*)         max_iter=$optarg ;;
         --reg-covar=*)        reg_covar=$optarg ;;
         --n-init=*)           n_init=$optarg ;;
         --fit-period-start=*) fit_period_start=$optarg ;;
         --fit-period-end=*)   fit_period_end=$optarg ;;
         --random-seed=*)      random_seed=$optarg ;;
         --verbose|-v)         verbose="yes" ;;
         --loss=*)             loss=$optarg ;;
         --pickle|-p)          pickle="yes" ;;
         --standardize|-s)     standardize="yes" ;;
         --standardize-by=*)   standardize_by=$optarg ;;
         --cross-validate)     cross_validate="yes" ;;
         --n-folds=*)          n_folds=$optarg ;;
         *) echo "Invalid option '$1'" ; exit 1 ;;
      esac

      shift

   done

fi

opts="\
--n-features $n_features \
--n-components $n_components \
--order $order \
--tol $tol \
--max-iter $max_iter \
--reg-covar $reg_covar \
--n-init $n_init \
--fit-period-start $fit_period_start \
--fit-period-end $fit_period_end \
--random-seed $random_seed \
--loss $loss \
"

if test ! "x$state_length" = "x" ; then
   opts="$opts --state-length $state_length"
fi

if test ! "x$presample_length" = "x" ; then
   opts="$opts --presample-length $presample_length"
fi

if test "x$verbose" = "xyes" ; then
   opts="$opts --verbose"
fi

if test "x$pickle" = "xyes" ; then
   opts="$opts --pickle"
fi

if test "x$standardize" = "xyes" ; then

   opts="$opts --standardize"

   if test ! "x$standardize_by" = "x" ; then
      opts="$opts --standardize-by $standardize_by"
   fi

fi

if test "x$cross_validate" = "xyes" ; then
   opts="$opts --cross-validate --n-folds $n_folds"
fi

data_basename="${VAR_NAME}.${TIMESPAN}.${CLIMATOLOGY_BASE_PERIOD}.anom"
data_basename="${data_basename}.${hemisphere}.${region}.${EOFS_BASE_PERIOD}.${season}"
data_basename="${data_basename}.max_eofs_${MAX_EOFS}.${LAT_WEIGHTS}.${EOFS_NORMALIZATION}"

input_file="${EOFS_NC_DIR}/${data_basename}.eofs.nc"

output_suffix="fembv_varx.n_pcs${n_features}.k${n_components}.m${order}"
if test ! "x$state_length" = "x" ; then
   output_suffix="${output_suffix}.state_length${state_length}"
else
   output_suffix="${output_suffix}.state_length0"
fi

output_file="${FEMBV_RESULTS_DIR}/${data_basename}.${output_suffix}.nc"

echo "Command: $PYTHON \"$FIT_MODEL\" $opts \"$input_file\" \"$output_file\""
$PYTHON "$FIT_MODEL" $opts "$input_file" "$output_file"
