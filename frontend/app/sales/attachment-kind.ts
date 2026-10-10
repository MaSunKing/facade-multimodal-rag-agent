export function attachmentKind(name: string): string {
  const extension = name.split('.').pop()?.toLowerCase();
  if (['png','jpg','jpeg','webp','gif','bmp','tif','tiff'].includes(extension || '')) return '图片';
  if (extension === 'pdf') return 'PDF';
  if (['doc','docx'].includes(extension || '')) return 'Word';
  if (['xls','xlsx','csv','tsv'].includes(extension || '')) return '表格';
  if (['ppt','pptx'].includes(extension || '')) return 'PPT';
  if (['txt','md','html','htm'].includes(extension || '')) return '文本';
  return '文件';
}
