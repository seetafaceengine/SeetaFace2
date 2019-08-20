#ifndef _SEETANET_PARSE_PROTO_HELPER_H
#define _SEETANET_PARSE_PROTO_HELPER_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

using std::vector;
using std::string;

int read( const char *buf, int len, int32_t &value );
int read( const char *buf, int len, uint32_t &value );
int read( const char *buf, int len, bool &value );
int read( const char *buf, int len, float &value );
int read( const char *buf, int len, vector<float> &value );
int read( const char *buf, int len, vector<uint32_t> &value );
int read( const char *buf, int len, vector<int32_t> &value );
int read( const char *buf, int len, std::string &value );
int read( const char *buf, int len, vector<string> &value );

int write( char *buf, int len, int32_t value );
int write( char *buf, int len, uint32_t value );
int write( char *buf, int len, bool value );
int write( char *buf, int len, float value );
int write( char *buf, int len, const vector<float> &value );
int write( char *buf, int len, const vector<uint32_t> &value );
int write( char *buf, int len, const vector<int32_t> &value );

int write( char *buf, int len, const std::string &value );
int write( char *buf, int len, const vector<string> &value );


int WriteStringToFile( const std::string &input_string, std::fstream &outputfstream );
int WriteStringVectorToFile( const std::vector<std::string> &vec, std::fstream &outputfstream );


#endif
