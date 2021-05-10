

mean = squeeze(nanmean(data)) ; 

for i = 1 : length(data)
    Mdata(i,:,:) = squeeze(data(i,:,:)) - mean;
end


[t1,t2,t3] = Dates(2013,2018,1:12) ;
t1(end) = [] ;
ssh(1:24,:,:) = [];
[t,at,on] = size(data) ;

% var_trend = nan(at,on) ;

for i = 1 : at
    for j = 1 : on
        a = polyfit(t1,squeeze(data(:,i,j)),1);
        A = polyval(a,t1) ;
        var_trend(i,j) = (A(end) - A(1));%./(t1(end)-t1(1)) ;     
       
    end
end

var_trend(var_trend>-10) =nan ;


mlon = lon_tr(1); Mlon = lon_tr(end) ; mlat = lat_tr(1) ; Mlat = lat_tr(end) ; 

lat_co = find(lat>=mlat & lat <= Mlat) ;
lon_co = find(lon>=mlon & lon <= Mlon) ;

