#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define die(...) do{printf(__VA_ARGS__); fputc('\n',stdout); exit(EXIT_FAILURE);}while(0);
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

void
debug_print_str(safetensors_Str s){
	for(int i=0; i < s.len; i++) fputc(s.ptr[i], stdout);
}

void 
debug_print_kv_str(safetensors_Str key, safetensors_Str value)
{
	debug_print_str(key);
	printf(" = ");
	debug_print_str(value);
	fputc('\n', stdout);
}

void 
debug_print_kv_intlist(safetensors_Str key, IntList *intlist)
{
	debug_print_str(key);
	printf(" = [");
	for(int i=0; i < intlist->num_entries; i++) printf("%lli, ", (long long) intlist->entries[i]);
	printf("]\n");
}


void* read_file(char *filename, int64_t *file_size)
{
	FILE *f = fopen(filename, "rb");

	if (!f)  die("can't open %s", filename);

	if(fseek(f, 0, SEEK_END)) die("can't fseek end on %s", filename);
	
	#ifdef _WIN32 
	#define FTELL(x) _ftelli64(x)
	#else
	#define FTELL(x) ftell(x)
	#endif
	
	int64_t pos = FTELL(f);
	
	if(pos == -1LL) die("invalid file size");
	*file_size = pos;
	if(fseek(f, 0, SEEK_SET)) die("can't fseek start on %s", filename);

	void *buffer = malloc(*file_size);
	if(!buffer) die("Can't malloc %lli bytes", (long long) *file_size);

	if(*file_size != (int64_t)fread(buffer, 1, *file_size, f)) die("cant fread");

	fclose(f);
	return buffer;
}


int main (int argc, char *argv[])
{
	char *filename = argv[1];
	if (!filename) return 0;

	int64_t sz = 0;
	void * file = read_file(filename, &sz);
	
	safetensors_File f = {0};

	char * result = safetensors_file_init(file, sz, &f); 
	if(result) {
		printf("%s\n", result);
		for ( char *s = MAX(file, f.error_context-20);
			s < MIN(f.error_context+21, f.one_byte_past_end_of_header);
			s++) fputc(*s,stdout);
		printf("\n");
		printf("                    ^ HERE\n");
		return 1;
	}

	printf("--- TENSORS\n\n");

	for(int i = 0; i < f.num_tensors; i++) {
		safetensors_TensorDescriptor t = f.tensors[i];
		debug_print_str(t.name);
		printf("\n\tdtype: %i\n\tshape: (%i) [", t.dtype, t.n_dimensions);
		for(int j = 0; j < t.n_dimensions; j++) {
			char *delim = j==t.n_dimensions-1 ? "" : ", ";
			printf("%lli%s", (long long) t.shape[j], delim);
		}
		printf("]\n\toffsets: [%lli, %lli]\n\tpointer: %p\n\n", 
				t.begin_offset_bytes, 
				t.end_offset_bytes,
				t.ptr);
	}

	printf("--- METADATA\n\n");

	for(int i = 0; i < f.num_metadata; i++) {
		safetensors_MetadataEntry m = f.metadata[i];
		debug_print_kv_str(m.name, m.value);
	}
}

