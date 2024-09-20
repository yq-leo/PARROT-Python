python main.py --dataset=Douban --record --exp_name=attr_noise
for attr_noise in $(seq 0.1 0.1 0.9); do
  python main.py --dataset=Douban --record --exp_name=attr_noise --attr_noise=$attr_noise --runs=10 --robust --strong_noise
done