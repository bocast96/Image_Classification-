function [nnHist, leHist, wds] = processing(imgs, k, clusters)
    
    %% 1.1.1 descriptors
    imgCount = size(imgs, 2);
    ds = [];
    ptsIm = zeros(imgCount,1);
    tmp = 0;
    for i = 1:imgCount
        im = rgb2gray(imgs{1,i});
        [~,d] = vl_sift(im);
        ds = [ds,d];
        tmp = tmp + size(d,2);
        ptsIm(i) = tmp;
    end
    ds = im2double(ds);

    %% 1.1.2 kmeans & encodings
    if nargin == 2
        % 1.1.2
        [wds,nn] = vl_kmeans(ds, k);
        featPointCnt = length(nn);
        
        % 1.1.3 local encoding
        localEnc = zeros(featPointCnt, k);

        for i = 1:featPointCnt
            im = ds(:,i);
            tmpR = zeros(1,k);
            for j = 1:k
                clst = wds(:,j);
                tmp = (clst - im).^2;
                tmp = sum(tmp);

                tmpR(j) = 1/tmp;
            end
            %tmpR = tmpR./sum(tmpR);
            localEnc(i,:) = tmpR;
        end
    
    else 
        wds = clusters;
        featPointCnt = size(ds,2);
        
        localEnc = zeros(featPointCnt, k);
        nn = zeros(1,featPointCnt);

        for i = 1:featPointCnt
            im = ds(:,i);
            tmpR = zeros(1,k);
            for j = 1:k
                clst = wds(:,j);
                tmp = (clst - im).^2;
                tmp = sum(tmp);

                tmpR(j) = 1/tmp;
            end
            [~,idx] = max(tmpR);
            nn(i) = idx;
            %tmpR = tmpR./sum(tmpR);
            localEnc(i,:) = tmpR;
        end 
    end

    %% nn histogram
    nnHist = zeros(imgCount,k);
    st = 1;
    for i = 1:imgCount
        ed = ptsIm(i);
        tmp = nn(1,st:ed);
        %h = histcounts(tmp, k, 'Normalization','probability');
        h = zeros(1,k);
        for j = 1:length(tmp)
            idx = tmp(j); 
            h(idx) = h(idx)+1;
        end
        h = h./sum(h);
        nnHist(i,:) = h;
        st = ed+1;
    end

    %% local encoding hist
    st = 1;
    leHist = zeros(imgCount,k);
    for i = 1:imgCount
        ed = ptsIm(i);
        tmp = localEnc(st:ed,:);
        tmp = mean(tmp);
        tmp = tmp./sum(tmp);
        leHist(i,:) = tmp;
        st = ed+1;
    end
end