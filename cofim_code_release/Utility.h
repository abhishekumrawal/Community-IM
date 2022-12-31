#pragma once

#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<string>
#include<iostream>

#define NUM_SIMUS 10000

using namespace std;

string currentTimestampStr() {
    time_t t = time(0);   // get time now
    struct tm * now = localtime( & t );
    char buf[1000];
    sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d", now->tm_year+1900, now->tm_mon+1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
    return string(buf);
}

void ExitMessage(string msg){
    cout << msg << endl;
    exit(1);
}

void randomOrder(int *rand_arr, int n){
	for(int i = 0; i < n; i++)
		rand_arr[i] = i;
	for(int i = 0; i < n; i++){
		int rand_pos = rand() % n;
		int tmp = rand_arr[i];
		rand_arr[i] = rand_arr[rand_pos];
		rand_arr[rand_pos] = tmp;
	}
}
