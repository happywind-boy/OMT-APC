function SaveEPS(FileName)
FileName = CheckFileName(FileName, 'eps');
export_fig(FileName);

function FileName = CheckFileName(FileName, Ext)
Ext_len = length(Ext);
if length(FileName) <= Ext_len+1
    FileName = [FileName '.' Ext];
else
    if ~strcmpi(FileName(end-Ext_len:end), ['.' Ext])
        FileName = [FileName '.' Ext];
    end
end