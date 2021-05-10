% confirm the cnv_file name !!!

fid = fopen(cnv_file);     % open file to read
fseek(fid,0,-1);                % set read position to beginning of file
while strcmp(fgetl(fid),'*END*') == 0 end        % go through lines until '*END*'

n=1;                            
while 1
    tline = fgetl(fid)   ;              % read in line
    if ~ischar(tline), break, end       % if eof, break and finish
    data(:,n) = sscanf(tline,'%f') ;    % put numbers in a matrix (in columns)
    n=n+1;
end

fclose(fid)
data = data';