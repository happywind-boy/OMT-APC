function SavePNG(FileName)
FileName = CheckFileName(FileName, 'png');
export_fig(FileName, '-transparent');

function FileName = CheckFileName(FileName, Ext)
Ext_len = length(Ext);
if length(FileName) <= Ext_len+1
    FileName = [FileName '.' Ext];
else
    if ~strcmpi(FileName(end-Ext_len:end), ['.' Ext])
        FileName = [FileName '.' Ext];
    end
end