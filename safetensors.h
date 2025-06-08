/*
	Harris M. Snyder, 2023
	This is free and unencumbered software released into the public domain.

	safetensors.h: a library for reading .safetensors files from C.

	Basic usage: read the entire .safetensors file into memory* (this is not
	handled by safetensors.h) and feed it to safetensors_file_init(). This
	will populate a safetensors_File struct, which contains an array of 
	tensor descriptors. You can then loop over the tensor descriptors and 
	pull out what you need. See the structs and functions below for details.

	*If you can't (or don't want to) read the whole file into memory, you can
	use safetensors_read_le_u64 to read the first 8 bytes of the file. This
	will tell you how big the header is. You can then read only that portion
	of the file and are guaranteed to get the entire header. It's safe to call 
	safetensors_file_init() on a buffer that contains at least that many 
	bytes.

	This file is a single-header library (credit to Sean Barrett for the
	idea); it includes both the header and the actual definitions in 
	a single file. To use this library, copy it  into your project, and 
	define SAFETENSORS_IMPLEMENTATION in exactly one .c file, immediately
	before you include safetensors.h

	The library depends only on the following headers from the standard
	library:
	  -  limits.h
	  -  stdint.h
	  -  stddef.h
	  -  stdlib.h
	The latter is for realloc. A future update will allow the user to 
	control the memory allocation, so that stdlib.h is not needed. 

	As of 2025-05-30, safetensors is defined to be little-endian. This 
	library will correctly parse a safetensors header on a host of either
	endianness, and safetensors_le_to_host can be used to convert tensor
	data from LE to host endianness.

*/

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#ifndef SAFETENSORS_API
#define SAFETENSORS_API
#endif

#ifndef SAFETENSORS_MAX_DIM 
#define SAFETENSORS_MAX_DIM 20 
#endif

#ifdef NONSTD_H
// compatibility with strings from https://github.com/hsnyder/nonstd
typedef struct Str safetensors_Str;
#else
typedef struct safetensors_Str {
	char *ptr;
	int len;
} safetensors_Str;
#endif

typedef struct {
	safetensors_Str name;
	// the pointer inside this struct will point into the 
	// memory block passed to safetensors_file_init()

	int dtype;
	// will be one of the enum values below
	
	int n_dimensions;
	int64_t shape[SAFETENSORS_MAX_DIM];
	// only the first n_dimensions entry of shape are meaningful 
	
	int64_t begin_offset_bytes;
	int64_t end_offset_bytes;
	// values taken directly from file. an offset of 0 means the
	// exact start of the portion of the file that follows the 
	// header (i.e. it is NOT an offset into the entire file).

	void *ptr;
	// this will be pre-populated assuming that the memory block 
	// that was fed to safetensors_file_init() was the entire file,
	// i.e. that the actual data immediately follows the header.
	// if this is not the case, this pointer will be bogus, use the
	// offsets to manually compute the location.
} safetensors_TensorDescriptor;

typedef struct {
	safetensors_Str name;
	safetensors_Str value;
} safetensors_MetadataEntry;

typedef struct {
	int c; // internal use

	int total_header_size;
	// offset within the file at which the data segment starts.
	// if you add begin_offset_bytes from a given tensor to this value,
	// you'll have the tensor's absolute offset into the file.

	char * error_context;
	// if safetensors_file_init() fails, this pointer will be set to 
	// where in the file memory block the error occurred.

	void * one_byte_past_end_of_header;
	// after calling safetensors_file_init, this will point to the 
	// next byte after the end of the header. If you've read
	// the whole file into memory, this is the start of the data segment.

	safetensors_TensorDescriptor *tensors;
	safetensors_MetadataEntry    *metadata;
	// these ^ are allocated automatically by safetensors_file_init()
	// and are contiguous arrays. the user should free() them when done.
	
	int num_tensors;
	int num_metadata;
	// the lengths of the above arrays
} safetensors_File;

SAFETENSORS_API char * safetensors_file_init(void *file_buffer, int64_t file_buffer_size_bytes, safetensors_File *out);
// Given a file buffer, parses the safetensors header and populates a safetensors_File 
// structure so that the client program can find the data it wants. file_buffer should 
// point to a buffer that contains at least the entire header, or more preferably the 
// whole safetensors file.
//
// Returns 0 on success. On failure, returns a static error message string and sets 
// out->error_context such that it points to where in file_buffer the error happened.


SAFETENSORS_API int safetensors_str_equal(safetensors_Str a, const char * b);
// For convenience: easily check if a tensor name matches a given string literal

SAFETENSORS_API int safetensors_lookup(safetensors_File *f, const char *name);
// For convenience: loop over tensors and return the index of the tensor whose 
// name matches a given string (or -1, if no match is found).


SAFETENSORS_API uint64_t safetensors_read_le_u64(uint8_t bytes[8]) ;
// Interpret 8 bytes as a little-endian uint64_t. Useful for reading the header
// size field at the very start of the safetensors file.

// Enum values for the 'dtype' field
enum {
	SAFETENSORS_F64 = 0,
	SAFETENSORS_F32,
	SAFETENSORS_F16,
	SAFETENSORS_BF16,
	SAFETENSORS_I64,
	SAFETENSORS_I32,
	SAFETENSORS_I16,
	SAFETENSORS_I8,
	SAFETENSORS_U8,
	SAFETENSORS_BOOL,
	SAFETENSORS_F8_E4M3,
	SAFETENSORS_F8_E5M2,
	
	SAFETENSORS_NUM_DTYPES
};

SAFETENSORS_API int safetensors_dtype_size(int dtype);
// For convenience: sizes of a given dtype code

SAFETENSORS_API const char *safetensors_dtype_name(int dtype);
// For convenience: human-readable names given a dtype code


SAFETENSORS_API void safetensors_le_to_host(void *data, ptrdiff_t data_len_bytes, int element_size);
// In-place byte order conversion from little-endian to host byte order (no-op on LE systems)
// Internally asserts that element_size is 1, 2, 4, or 8.

//SAFETENSORS_API void safetensors_host_to_le(void *data, ptrdiff_t data_len);
//// In-place: makes sure data is in little-endian order, for safetensors writing
//TODO: add support for writing safetensors files


#endif

/* 
   ============================================================================
		END OF HEADER SECTION
		Implementation follows
   ============================================================================
*/


#ifdef SAFETENSORS_IMPLEMENTATION

#ifndef assert
#  ifdef SAFETENSORS_DISABLE_ASSERTIONS
#    define assert(c)
#  else
#    if defined(_MSC_VER)
#      define assert(c) if(!(c)){__debugbreak();}
#    else
#      if defined(__GNUC__) || defined(__clang__)
#        define assert(c) if(!(c)){__builtin_trap();}
#      else 
#        define assert(c) if(!(c)){*(volatile int*)0=0;}
#      endif 
#    endif
#  endif
#endif

#include <limits.h>
#include <stdlib.h>


static int safetensors_host_is_little_endian(void)
{
	union {
		unsigned int i;
		unsigned char c[sizeof(unsigned int)];
	} test;
	test.i = 1;
	return test.c[0] == 1;
}

static void safetensors_bswap(uint8_t* p, int size) 
{
	// despite appearances, gcc14 does a good job at -O3 on s390x.
	for (int i = 0; i < size / 2; ++i) {
		uint8_t tmp = p[i];
		p[i] = p[size - 1 - i];
		p[size - 1 - i] = tmp;
	}
}


void
safetensors_le_to_host(void *data, ptrdiff_t data_len_bytes, int element_size)
{       
        if(safetensors_host_is_little_endian()) return;
        if(element_size == 1) return;
        
        assert(element_size == 2 || element_size == 4 || element_size == 8);

        uint8_t *p = (uint8_t*)data;
        data_len_bytes = (data_len_bytes/element_size)*element_size;

        for(ptrdiff_t i = 0; i < data_len_bytes; i+=element_size) {
                if(element_size==2)       safetensors_bswap(p+i, 2);
                else if (element_size==4) safetensors_bswap(p+i, 4);
                else if (element_size==8) safetensors_bswap(p+i, 8);
        }
}


static int safetensors_strlen(const char *b) 
{
	if(!b) return 0;
	int i = 0;
	while (*b != 0) {
		i++; b++;
	}
	return i;
}

SAFETENSORS_API int safetensors_str_equal(safetensors_Str a, const char * b)
// For convenience: easily check if a tensor name matches a given string literal
{
	if (!b) return 0;
	if (a.len==0) return 0;
	if (safetensors_strlen(b) != a.len) return 0;
	int equal = 1;
	for (int i = 0  ;  (i < a.len && equal && b[i])  ;  i++) 
		equal = equal  &&  a.ptr[i] == b[i];
	return equal;
}

SAFETENSORS_API int safetensors_lookup(safetensors_File *f, const char *name)
// For convenience: loop over tensors and return the index of the tensor whose 
// name matches a given string (or -1, if no match is found).
{
	for(int i = 0; i < f->num_tensors; i++)
		if(safetensors_str_equal(f->tensors[i].name, name))
			return i;
	return -1;
}

// For convenience: sizes of a given dtype code
SAFETENSORS_API int safetensors_dtype_size(int dtype)
{
	switch(dtype) {
	case SAFETENSORS_F64:  return 8;
	case SAFETENSORS_F32:  return 4;
	case SAFETENSORS_F16:  return 2;
	case SAFETENSORS_BF16: return 2;
	case SAFETENSORS_I64:  return 8;
	case SAFETENSORS_I32:  return 4;
	case SAFETENSORS_I16:  return 2;
	case SAFETENSORS_I8:   return 1;
	case SAFETENSORS_U8:   return 1;
	case SAFETENSORS_BOOL: return 1; // TODO check if this is right
	case SAFETENSORS_F8_E4M3: return 1;
	case SAFETENSORS_F8_E5M2: return 1;
	}
	return 0;
}


// For convenience: human-readable names given a dtype code
SAFETENSORS_API const char *safetensors_dtype_name(int dtype)
{
	static const char *sft_type_names[] = {
	    [SAFETENSORS_F64]  = "F64",
	    [SAFETENSORS_F32]  = "F32",
	    [SAFETENSORS_F16]  = "F16",
	    [SAFETENSORS_BF16] = "BF16",
	    [SAFETENSORS_I64]  = "I64",
	    [SAFETENSORS_I32]  = "I32",
	    [SAFETENSORS_I16]  = "I16",
	    [SAFETENSORS_I8]   = "I8",
	    [SAFETENSORS_U8]   = "U8",
	    [SAFETENSORS_BOOL] = "BOOL",	
	    [SAFETENSORS_F8_E4M3] = "F8_E4M3",
	    [SAFETENSORS_F8_E5M2] = "F8_E5M2",
	};

	if (dtype >= 0 && dtype < (int)(sizeof(sft_type_names)/sizeof(sft_type_names[0])))
		return sft_type_names[dtype];

	return "INVALID";
}


static void *safetensors_default_realloc(void *p, ptrdiff_t len, void *ctx) {
	(void) ctx;
	return realloc(p,len);
}

typedef struct safetensors_Allocator {
	void *(*realloc)(void *, ptrdiff_t, void*) ;
	void *ctx;
} safetensors_Allocator;



static safetensors_Allocator safetensors_default_allocator(void) {
	safetensors_Allocator al;
	al.realloc = safetensors_default_realloc;
	al.ctx     = (void*)0;
	return al;
}

static int64_t
safetensors_parse_positive_int(char **ptr, char *limit)
{
	/*
	 	- Skips preceeding spaces and tabs
		- Won't read past 'limit'
		- Doesn't check for integer overflow
		- Doesn't parse negative numbers
		- Returns -1 on failure
	*/
	char * str = *ptr;

	while(*str == ' ' || *str == '\t') str++;

	int64_t v = 0;
	int n = 0;
	while (str < limit  &&  *str >= 48  &&  *str <= 57) 
	{
		int digit = *str - 48;
		v *= 10;
		v += digit;
		str++;
		n++;
	}

	if (n > 0) {
		*ptr = str;
		return v;
	}

	return -1;
}

static int
safetensors_eat(char **ptr, char *limit, char expected)
{
	// skip whitespace
	// return 0 if we hit the limit
	// return 0 if data don't match the expected string
	// otherwise, return 1 and move the pointer to the end of the match
	char *p = *ptr;
	while((p < limit) && (*p == ' ' || *p == '\t')) ++p;
	if (p + 1 > limit) return 0;
	if (*p != expected) return 0;
	*ptr = p + 1;
	return 1;
}

static int
safetensors_peek (char *ptr, char *limit, char expected)
{
	// same as eat, but doesn't adjust the pointer
	char *tmp = ptr;
	return safetensors_eat(&tmp, limit, expected);
}


typedef struct {
	int num_entries;
	int64_t entries[SAFETENSORS_MAX_DIM];
} safetensors_IntList;

static int 
safetensors_eat_intlist(char **ptr, char *limit, safetensors_IntList *out)
{
	*out = (safetensors_IntList){0};
	char *p = *ptr;
	if(!safetensors_eat(&p,limit,'[')) return 0;

	while (p < limit) {
		char *p_save = p;
		if(safetensors_eat(&p,limit,']')) 
			break;

		int64_t val = safetensors_parse_positive_int(&p,limit);
		if (val == -1) {
			return 0;
		} else {
			out->entries[out->num_entries++] = val;
			if (out->num_entries == SAFETENSORS_MAX_DIM) {
				return 0; // unsupported tensor dimensions (TODO improve handling)
			}
		}

		if(!safetensors_eat(&p, limit, ','))
			if(!safetensors_peek(p, limit, ']'))
				return 0;

		assert(p != p_save);
	}

	*ptr = p;
	return 1;
}


static int
safetensors_eat_string(char **ptr, char *limit, safetensors_Str *out) 
{
	char delim = 0; 

	if      (safetensors_eat(ptr, limit, '\'')) delim = '\'';
	else if (safetensors_eat(ptr, limit, '"' )) delim = '"';
	else return 0; // bad delimiter

	int len = 0;
	char *p = *ptr;
	char *start = p;

	while (p < limit) {
		if (*p == delim  &&  p[-1] != '\\') {
			++p; 
			goto string_ok;
		} else {
			++p; 
			++len;
		}
	}
	return 0; // unterminated
		      
	string_ok: assert(p <= limit);
	*ptr = p;
	safetensors_Str str;
	str.len = len;
	str.ptr = start;
	*out = str;
	return 1;
}

typedef struct {
	safetensors_Str key;
	int value_is_str;
	union {
		safetensors_Str     svalue;
		safetensors_IntList ivalue;
	};
} safetensors_KeyValuePair;

static int
safetensors_eat_kv_pair(char **ptr, char *limit, safetensors_KeyValuePair *kvp)
{
	char *p = *ptr;

	// mandatory string (key)
	if(!safetensors_eat_string(&p, limit, &kvp->key)) 
		return 0;

	if(!safetensors_eat(&p, limit, ':'))
		return 0;

	// value can be string, or list of integers
	safetensors_Str str_value = {0};
	safetensors_IntList intlist_value = {0};
	
	if (!safetensors_eat_string(&p, limit, &str_value)){
		if (!safetensors_eat_intlist(&p, limit, &intlist_value)){
			return 0;
		} else {
			kvp->value_is_str = 0;
			kvp->ivalue = intlist_value;
		}
	} else {
		kvp->value_is_str = 1;
		kvp->svalue = str_value;
	}

	*ptr = p;
	return 1;
}

static char *
safetensors_more_memory(safetensors_File *out)
{
	safetensors_Allocator al = safetensors_default_allocator();
	if(out->num_tensors == out->c || out->num_metadata == out->c) {
		void *new_tensors = al.realloc(out->tensors, sizeof(out->tensors[0])*(out->c+100), al.ctx);
		if (!new_tensors)
			return (char*)"Out of memory";
		out->tensors = (safetensors_TensorDescriptor*)new_tensors;

		void *new_metadata = al.realloc(out->metadata, sizeof(out->metadata[0])*(out->c+100), al.ctx);
		if (!new_metadata)
			return (char*)"Out of memory";
		out->metadata = (safetensors_MetadataEntry*)new_metadata;

		out->c += 100;
	}
	return 0;
}

static char *
safetensors_apply_key_value_pair(safetensors_File *out, safetensors_KeyValuePair kvp, char *baseptr)
{
	#define KNOWN_DTYPES "F64, F32, F16, BF16, I64, I32, I16, I8, U8, or BOOL"
	if (safetensors_str_equal(kvp.key, "dtype")) {
		if (!kvp.value_is_str)
			return (char*)"Expected a string value for 'dtype'";
		if (safetensors_str_equal(kvp.svalue, "F64"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_F64;
		else if (safetensors_str_equal(kvp.svalue, "F32"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_F32;
		else if (safetensors_str_equal(kvp.svalue, "F16"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_F16;
		else if (safetensors_str_equal(kvp.svalue, "BF16"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_BF16;
		else if (safetensors_str_equal(kvp.svalue, "I64"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_I64;
		else if (safetensors_str_equal(kvp.svalue, "I32"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_I32;
		else if (safetensors_str_equal(kvp.svalue, "I16"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_I16;
		else if (safetensors_str_equal(kvp.svalue, "I8"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_I8;
		else if (safetensors_str_equal(kvp.svalue, "U8"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_U8;
		else if (safetensors_str_equal(kvp.svalue, "BOOL"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_BOOL;
		else if (safetensors_str_equal(kvp.svalue, "F8_E4M3"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_F8_E4M3;
		else if (safetensors_str_equal(kvp.svalue, "F8_E5M2"))
			out->tensors[out->num_tensors].dtype = SAFETENSORS_F8_E5M2;
		else return (char*)"Unrecognized datatype (expected " KNOWN_DTYPES ")";

	} else if (safetensors_str_equal(kvp.key, "shape")) {
		if (kvp.value_is_str)
			return (char*)"Expected an integer list value for 'shape'";
		out->tensors[out->num_tensors].n_dimensions = kvp.ivalue.num_entries;
		for(int i = 0; i < kvp.ivalue.num_entries; i++)
			out->tensors[out->num_tensors].shape[i] = kvp.ivalue.entries[i];
	} else if (safetensors_str_equal(kvp.key, "data_offsets")) {
		if (kvp.value_is_str)
			return (char*)"Expected an integer list value for 'shape'";
		if (kvp.ivalue.num_entries != 2)
			return (char*)"Expected exactly two entries for the value of 'offsets'";
		out->tensors[out->num_tensors].begin_offset_bytes = kvp.ivalue.entries[0];
		out->tensors[out->num_tensors].end_offset_bytes   = kvp.ivalue.entries[1];
		out->tensors[out->num_tensors].ptr = baseptr + kvp.ivalue.entries[0];
	} else {
		// error? ignore?
		return (char*)"Unexpected key (expected dtype, shape, or data_offsets)";
	}
	return 0;
}


SAFETENSORS_API uint64_t 
safetensors_read_le_u64(uint8_t bytes[8]) {
    return ((uint64_t)bytes[0])       |
           ((uint64_t)bytes[1] << 8)  |
           ((uint64_t)bytes[2] << 16) |
           ((uint64_t)bytes[3] << 24) |
           ((uint64_t)bytes[4] << 32) |
           ((uint64_t)bytes[5] << 40) |
           ((uint64_t)bytes[6] << 48) |
           ((uint64_t)bytes[7] << 56);
}


SAFETENSORS_API char *
safetensors_file_init(void *file_buffer, int64_t file_buffer_bytes, safetensors_File *out)
{	
	if (file_buffer_bytes < 8) {return (char*) "Buffer < 8 bytes: cannot possibly be a valid safetensors file."; }

	*out = (safetensors_File){0};
	int header_len = 0;
	{
		uint64_t header_len_u64 = safetensors_read_le_u64((uint8_t*)file_buffer);
		#define STRINGIFY(x) #x
		if (header_len_u64 > (uint64_t)INT_MAX) 
			return (char*)"File header allegedly more than INT_MAX (" STRINGIFY(INT_MAX) ") bytes, file likely corrupt";
		header_len = header_len_u64;
	}
	assert(header_len >= 0);
	if (header_len == 0) 
		return (char*)"File header allegedly zero bytes, file likely corrupt";

	char *t = ((char*)file_buffer)+8;
	char *e = t + header_len;
	out->one_byte_past_end_of_header = e;
	out->total_header_size = header_len+8;

	char *tensor_data_baseptr = t + header_len;

	#define ST_ERR(message) return out->error_context = t, (char*)(message)

	// mandatory open brace starts the header
	if (!safetensors_eat(&t,e,'{')) ST_ERR("Expected '{'");

	// loop over header entries
	while (t<e) {
		char *t_save = t;

		// if we hit a close brace, we're done
		if (safetensors_eat(&t,e,'}')) goto header_ok;

		// mandatory string (tensor name)
		safetensors_Str tensor_name = {0};
		if (!safetensors_eat_string(&t,e,&tensor_name)) 
			ST_ERR("Expected tensor name");
		if (!safetensors_eat(&t,e,':'))
			ST_ERR("Expected colon after tensor name");

		char * alloc_error = safetensors_more_memory(out);
		if (alloc_error) ST_ERR(alloc_error);

		out->tensors[out->num_tensors].name = tensor_name;

		// open brace starts a header entry
		if (safetensors_eat(&t,e,'{')) {

			// loop over key-value pairs inside the header entry
			while (t<e) {
				char *t_save = 0;

				// close brace terminates the header entry
				if (safetensors_eat(&t,e,'}')) {
					if(!safetensors_str_equal(tensor_name, "__metadata__"))
						++out->num_tensors;
					break;
				}

				// otherwise it's a key-value pair
				safetensors_KeyValuePair kvp = {0};
				char *error_context = t;
				if(!safetensors_eat_kv_pair(&t,e,&kvp))
					ST_ERR("Expected a key-value pair");

				// figure out what to do with the key-value pair
				if(safetensors_str_equal(tensor_name, "__metadata__")) {
					if(!kvp.value_is_str) 
						return out->error_context=error_context, 
								(char*)("Expected a string value for a metadata entry");
					out->metadata[out->num_metadata++] =
						(safetensors_MetadataEntry) {
							.name  = kvp.key,
							.value = kvp.svalue
						};
				} else {
					char * kvp_error = safetensors_apply_key_value_pair(out,kvp,tensor_data_baseptr);
					if(kvp_error) return out->error_context=error_context, kvp_error;
				}

				if(!safetensors_eat(&t,e,','))
					if(!safetensors_peek(t,e,'}'))
						ST_ERR("Expected comma");

				assert(t != t_save);
			}
		}

		if(!safetensors_eat(&t,e,','))
			if(!safetensors_peek(t,e,'}'))
				ST_ERR("Expected comma");

		assert(t != t_save);
	}
	ST_ERR("Unterminated header");
	header_ok: return 0;
	#undef ST_ERR
}

#endif
