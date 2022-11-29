function profiling(){
	d="profiling_`date +%Y%m%d%H%M%S`"
	vtune -collect uarch-exploration -r ${d} -data-limit=1000000 ./build/sift1B_demo
}

profiling