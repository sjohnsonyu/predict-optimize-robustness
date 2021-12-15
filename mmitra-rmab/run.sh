USERNAME=$1
PASSWORD=$2
DATE=$3
INTERVENTIONS=$4
REG_DATE=$5

mkdir data
mkdir data/beneficiary
mkdir data/call
chmod 777 get_data.sh
./get_data.sh ${USERNAME} ${PASSWORD} ${DATE} ${REG_DATE}
pip install -r requirements.txt
unzip policy.zip
python rmab_individual_clustering.py ${DATE} ${INTERVENTIONS} 0 ${REG_DATE} ${USERNAME} ${PASSWORD}
rm -rf data
chmod 777 insert_interventions.sh
./insert_interventions.sh ${USERNAME} ${PASSWORD} ${DATE}
