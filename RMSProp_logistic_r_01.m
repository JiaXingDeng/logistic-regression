function[wt,t,s]=RMSProp_logistic_r_01(x,y,w0,u,a,e)
%%
%ʹ��RMSProp������logistic regression���˴�yΪ-1��1
% xָ����,yָ��Ӧ��ǩ��aָ�ݶ��½�ѧϰ�ʣ�uָ����������w0Ϊ��ʼֵ
% EXAMPLE��
% load('logistic_regression.mat');
%[wt,t,s]=RMSProp_logistic_r_n1_1(x,y,[0,0,0]',0.9,0.01,10^(-6))
l=length(y);
x1=[x,ones(l,1)];%���ݼ���
z=x1*w0;
s=sum(-log(exp(-y.*z)./(1+exp(-y.*z))));%��ʧ��������
t=0;
v=zeros(3,1);%������ֵΪ0
loss=[];
while s>0.00001 && t>30000
        gammax=sum(y.*sigmoid(-y.*z).*x1)';%�ݶȼ���
        vt=u*v+(1-u)*gammax.*gammax;%��������
        v=vt;
        wt=w0-a*(gammax./(v.^(1/2)+e));%��������
        w0=wt;
        z=x1*w0;
        s=sum(-log(exp(-y.*z)./(1+exp(-y.*z))));
        t=t+1;
        loss(t)=s;
end
%%
%�����ͼ
xp = x(y>0,:)';
xn = x(y==0,:)';
subplot(211)
plot(xp(1,:),xp(2,:),'bo','linewidth',1.5)
hold on
plot(xn(1,:),xn(2,:),'rx','linewidth',1.5)
grid on
p1 = 0;
p2 = (-wt(1)*p1-wt(3))/wt(2);
q1 = 12;
q2 = (-wt(1)*q1-wt(3))/wt(2);
plot([p1 q1],[p2 q2],'k-','linewidth',1.5)
set(gca,'position',[0.04 0.55 0.94 0.43])             
xlabel('Decision Boundary of Perceptron')
hold off;
subplot(212)
tt = 1:10:t-1;
plot(tt,loss(tt),'b-','linewidth',1.5);
xlabel('epoch')
ylabel('loss')
axis([0 9000 0 10])
set(gca,'position',[0.04 0.07 0.94 0.42]

    